import triton_python_backend_utils as pb_utils
import numpy as np
import re

class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.logger.log_info("Stitcher model initialized.")

    def _parse_asr_output(self, asr_lines):
        """Parses '[start] - [end]: text' into a structured list."""
        segments = []
        # Regex to capture start, end, and text
        pattern = re.compile(r'\[([\d.]+)\] - \[([\d.]+)\]: (.*)')
        for line in asr_lines:
            match = pattern.match(line)
            if match:
                segments.append({
                    "start": float(match.group(1)),
                    "end": float(match.group(2)),
                    "text": match.group(3).strip()
                })
        return segments

    def _parse_rttm_output(self, rttm_lines):
        """Parses RTTM lines into a structured list."""
        speaker_turns = []
        for line in rttm_lines:
            parts = line.split()
            if parts[0] == 'SPEAKER':
                start = float(parts[3])
                duration = float(parts[4])
                speaker_turns.append({
                    "start": start,
                    "end": start + duration,
                    "speaker": parts[7]
                })
        # Sort by start time
        speaker_turns.sort(key=lambda x: x['start'])
        return speaker_turns

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                # Get inputs
                asr_tensor = pb_utils.get_input_tensor_by_name(request, "ASR_TRANSCRIPT")
                asr_lines = [line.decode('utf-8') for line in asr_tensor.as_numpy()]

                rttm_tensor = pb_utils.get_input_tensor_by_name(request, "DIARIZATION_RTTM")
                rttm_lines = [line.decode('utf-8') for line in rttm_tensor.as_numpy()]

                # Parse inputs
                asr_segments = self._parse_asr_output(asr_lines)
                speaker_turns = self._parse_rttm_output(rttm_lines)
                
                final_transcript = []

                if not speaker_turns: 
                    if asr_segments:
                         final_transcript = [f"[{s['start']:.2f}] UNKNOWN: {s['text']}" for s in asr_segments]
                    else:
                         final_transcript = ["[No speech detected or diarization failed.]"]
                else:
                    # --- LOGIC FIX (UPDATED): Lọc Hallucination chặt chẽ hơn ---
                    # Lấy mốc thời gian kết thúc tuyệt đối của Diarizer
                    last_speech_end_time = speaker_turns[-1]['end']

                    final_segments = []
                    for seg in asr_segments:
                        seg_midpoint = seg['start'] + (seg['end'] - seg['start']) / 2
                        
                        # LOGIC MỚI: Kiểm tra Midpoint
                        # Nếu điểm giữa của đoạn text nằm sau thời điểm kết thúc hội thoại (cộng dư 0.5s)
                        # thì coi là ảo giác -> Bỏ qua ngay lập tức.
                        if seg_midpoint > last_speech_end_time + 0.5:
                            continue
                        
                        assigned_speaker = "UNKNOWN"
                        
                        # Tìm người nói dựa trên midpoint
                        for turn in speaker_turns:
                            if turn['start'] <= seg_midpoint < turn['end']:
                                assigned_speaker = turn['speaker']
                                break
                        
                        # Nếu vẫn là UNKNOWN và đoạn này nằm ở cuối file (sau speaker cuối cùng)
                        # thì khả năng cao vẫn là rác -> Bỏ qua
                        if assigned_speaker == "UNKNOWN" and seg['start'] > speaker_turns[-1]['start']:
                             continue

                        final_segments.append({
                            "start": seg['start'],
                            "end": seg['end'],
                            "speaker": assigned_speaker,
                            "text": seg['text']
                        })

                    # Combine consecutive segments
                    merged_transcript = []
                    if final_segments:
                        current_speaker = final_segments[0]['speaker']
                        current_text = f"[{final_segments[0]['start']:.2f}] {final_segments[0]['speaker']}: {final_segments[0]['text']}"
                        
                        for i in range(1, len(final_segments)):
                            seg = final_segments[i]
                            if seg['speaker'] == current_speaker:
                                current_text += " " + seg['text']
                            else:
                                merged_transcript.append(current_text)
                                current_speaker = seg['speaker']
                                current_text = f"[{seg['start']:.2f}] {seg['speaker']}: {seg['text']}"
                        merged_transcript.append(current_text)
                    
                    if merged_transcript:
                        final_transcript = merged_transcript
                    elif not final_transcript and asr_segments:
                        final_transcript = ["[No valid speech segments matched diarization timestamps.]"]

                # Create output tensor
                output_tensor = pb_utils.Tensor(
                    "FINAL_TRANSCRIPT",
                    np.array(final_transcript, dtype=object)
                )
                
                responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

            except Exception as e:
                import traceback
                self.logger.log_error(traceback.format_exc())
                error = pb_utils.TritonError(message=f"Error in stitcher: {str(e)}")
                responses.append(pb_utils.InferenceResponse(output_tensors=[], error=error))
        return responses

    def finalize(self):
        self.logger.log_info("Stitcher model finalized.")
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
            # RTTM format: SPEAKER <file> <chnl> <start> <duration> <NA> <NA> <speaker> <NA> <NA>
            if parts[0] == 'SPEAKER':
                start = float(parts[3])
                duration = float(parts[4])
                speaker_turns.append({
                    "start": start,
                    "end": start + duration,
                    "speaker": parts[7]
                })
        # Sort by start time just in case
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
                
                if not speaker_turns: # Handle case with no detected speech
                    final_transcript = ["[No speech detected or diarization failed.]"]
                else:
                    # Stitching logic: Assign a speaker to each ASR segment
                    final_segments = []
                    for seg in asr_segments:
                        seg_midpoint = seg['start'] + (seg['end'] - seg['start']) / 2
                        assigned_speaker = "UNKNOWN"
                        
                        # Find which speaker turn the segment's midpoint falls into
                        for turn in speaker_turns:
                            if turn['start'] <= seg_midpoint < turn['end']:
                                assigned_speaker = turn['speaker']
                                break
                        
                        final_segments.append({
                            "start": seg['start'],
                            "end": seg['end'],
                            "speaker": assigned_speaker,
                            "text": seg['text']
                        })

                    # Combine consecutive segments from the same speaker
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
                    
                    final_transcript = merged_transcript

                # Create output tensor
                output_tensor = pb_utils.Tensor(
                    "FINAL_TRANSCRIPT",
                    np.array(final_transcript, dtype=object)
                )
                
                responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

            except Exception as e:
                error = pb_utils.TritonError(message=f"Error in stitcher: {str(e)}")
                responses.append(pb_utils.InferenceResponse(output_tensors=[], error=error))
        return responses

    def finalize(self):
        self.logger.log_info("Stitcher model finalized.")
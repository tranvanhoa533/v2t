import triton_python_backend_utils as pb_utils
import numpy as np
import json

class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.logger.log_info("Stitcher (Word-Level) model initialized.")

    def _parse_rttm_output(self, rttm_lines):
        """Parses RTTM lines into a structured list."""
        speaker_turns = []
        for line in rttm_lines:
            parts = line.split()
            if len(parts) > 7 and parts[0] == 'SPEAKER':
                try:
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker_turns.append({
                        "start": start,
                        "end": start + duration,
                        "speaker": parts[7]
                    })
                except ValueError as e:
                    self.logger.log_warn(f"Stitcher: Error parsing RTTM line '{line}': {e}")

        speaker_turns.sort(key=lambda x: x['start'])
        return speaker_turns

    def execute(self, requests):
        self.logger.log_info(f"Stitcher: Received {len(requests)} requests.")
        responses = []
        for request in requests:
            try:
                # Get inputs
                asr_tensor = pb_utils.get_input_tensor_by_name(request, "ASR_TRANSCRIPT")
                # ASR bây giờ trả về 1 JSON string chứa list các từ
                asr_json_bytes = asr_tensor.as_numpy()[0]
                
                rttm_tensor = pb_utils.get_input_tensor_by_name(request, "DIARIZATION_RTTM")
                rttm_lines = [line.decode('utf-8') for line in rttm_tensor.as_numpy()]

                # --- LOG DEBUG: Inputs ---
                self.logger.log_info(f"Stitcher: Raw ASR JSON size: {len(asr_json_bytes)} bytes")
                self.logger.log_info(f"Stitcher: Received {len(rttm_lines)} RTTM lines.")

                # Parse inputs
                words = []
                try:
                    # Decode bytes -> string -> json list
                    asr_json_str = asr_json_bytes.decode('utf-8')
                    words = json.loads(asr_json_str)
                    self.logger.log_info(f"Stitcher: Successfully parsed {len(words)} words from ASR.")
                    if len(words) > 0:
                        self.logger.log_info(f"Stitcher: First word: {words[0]}, Last word: {words[-1]}")
                except Exception as e:
                    self.logger.log_error(f"Stitcher: Failed to parse ASR JSON: {e}")
                    # Fallback nếu ASR trả về format cũ hoặc rỗng
                    words = []

                speaker_turns = self._parse_rttm_output(rttm_lines)
                self.logger.log_info(f"Stitcher: Parsed {len(speaker_turns)} valid speaker turns.")
                
                final_transcript = []

                if not speaker_turns:
                    self.logger.log_warn("Stitcher: No speaker turns found from Diarizer.")
                    # Fallback: In ra text không có speaker
                    full_text = " ".join([w.get('word', '') for w in words])
                    if full_text:
                        final_transcript = [f"[0.00] UNKNOWN: {full_text}"]
                    else:
                        final_transcript = ["[No speech detected]"]
                else:
                    # --- Word-Level Stitching Logic ---
                    current_segment_speaker = None
                    current_segment_words = []
                    current_segment_start = 0.0
                    
                    last_speech_end_time = speaker_turns[-1]['end']
                    self.logger.log_info(f"Stitcher: Last speech timestamp from Diarizer: {last_speech_end_time:.3f}s")

                    for i, word in enumerate(words):
                        word_text = word.get('word', '')
                        start_t = word.get('start', 0.0)
                        end_t = word.get('end', 0.0)
                        
                        # 1. Lọc Hallucination (như cũ)
                        word_midpoint = (start_t + end_t) / 2
                        
                        # LOG DEBUG chi tiết cho từng từ (có thể comment lại nếu quá nhiều log)
                        # self.logger.log_info(f"Processing word '{word_text}' ({start_t:.2f}-{end_t:.2f}), Mid: {word_midpoint:.2f}")

                        if word_midpoint > last_speech_end_time + 0.5:
                            self.logger.log_info(f"Stitcher: Dropping hallucination word '{word_text}' at {start_t:.2f}s ( > {last_speech_end_time:.2f}s)")
                            continue

                        # 2. Tìm Speaker cho TỪNG TỪ
                        assigned_speaker = "UNKNOWN"
                        for turn in speaker_turns:
                            # Cho phép sai số nhỏ (buffer) để bắt từ nằm ngay biên
                            if turn['start'] - 0.2 <= word_midpoint <= turn['end'] + 0.2:
                                assigned_speaker = turn['speaker']
                                break
                        
                        # Nếu vẫn UNKNOWN và đây là những từ đầu tiên/cuối cùng, thử gán nới lỏng hơn
                        if assigned_speaker == "UNKNOWN":
                             # Logic fallback đơn giản: Gán cho speaker gần nhất nếu khoảng cách < 0.5s
                             pass 

                        # Nếu từ này thuộc speaker khác với segment đang gom -> Ngắt dòng
                        if assigned_speaker != current_segment_speaker:
                            self.logger.log_info(f"Stitcher: Switch Speaker detected at word '{word_text}' ({start_t}s). Old: {current_segment_speaker}, New: {assigned_speaker}")
                            
                            if current_segment_words:
                                # Đẩy segment cũ vào kết quả
                                text = " ".join(current_segment_words)
                                line = f"[{current_segment_start:.2f}] {current_segment_speaker}: {text}"
                                final_transcript.append(line)
                                # self.logger.log_info(f"Stitcher: Added line: {line}")
                            
                            # Bắt đầu segment mới
                            current_segment_speaker = assigned_speaker
                            current_segment_words = [word_text]
                            current_segment_start = start_t
                        else:
                            # Cùng speaker -> gom tiếp
                            current_segment_words.append(word_text)

                    # Đẩy segment cuối cùng
                    if current_segment_words:
                         text = " ".join(current_segment_words)
                         line = f"[{current_segment_start:.2f}] {current_segment_speaker}: {text}"
                         final_transcript.append(line)
                         self.logger.log_info(f"Stitcher: Added final line: {line}")

                # Create output tensor
                output_tensor = pb_utils.Tensor(
                    "FINAL_TRANSCRIPT",
                    np.array(final_transcript, dtype=object)
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.logger.log_error(f"Stitcher Error: {str(e)}\n{tb}")
                error = pb_utils.TritonError(message=f"Error: {str(e)}")
                responses.append(pb_utils.InferenceResponse(output_tensors=[], error=error))
        
        return responses

    def finalize(self):
        pass
import json
import os
import triton_python_backend_utils as pb_utils
import numpy as np
import nemo.collections.asr as nemo_asr
import torch
import tempfile
import soundfile as sf
import io

class TritonPythonModel:
    def initialize(self, args):
        """
        Initialize the model and load the local NeMo diarization model file.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = pb_utils.Logger

        model_repository = args['model_repository']
        model_version = args['model_version']
        model_dir = os.path.join(model_repository, model_version, 'checkpoints')

        nemo_file_name = 'diar_streaming_sortformer_4spk-v2.nemo'
        local_model_path = os.path.join(model_dir, nemo_file_name)

        if not os.path.exists(local_model_path):
            raise pb_utils.TritonModelException(
                f"Nemo model file not found at: {local_model_path}. "
                f"Please make sure '{nemo_file_name}' is in the '{model_dir}' directory."
            )

        self.logger.log_info(f"Loading local NeMo model from: {local_model_path}")

        # Load the Sortformer model
        self.diarizer_model = nemo_asr.models.EncDecDiarLabelModel.restore_from(
            restore_path=local_model_path,
            map_location=self.device
        )
        self.diarizer_model.eval()  # Set to evaluation mode

        self.logger.log_info("Local NeMo Sortformer Diarizer model initialized successfully.")

    def execute(self, requests):
        """
        This function is called for each inference request.
        """
        responses = []
        for request in requests:
            try:
                audio_bytes_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_BYTES")
                audio_bytes = audio_bytes_tensor.as_numpy()[0]

                # Assume the incoming audio is 16-bit signed integer PCM, at 16000 Hz
                sample_rate = 16000
                dtype = np.int16

                # Convert raw bytes to a NumPy array
                audio_data = np.frombuffer(audio_bytes, dtype=dtype)
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
                    # Write the NumPy array to a temporary file. soundfile adds the WAV header.
                    sf.write(tmp_audio_file.name, audio_data, sample_rate)
                    tmp_filename = tmp_audio_file.name
                    
                try:
                    self.logger.log_info(f"Processing audio file for diarization: {tmp_filename}")
                    
                    # Use the diarize() method with the correct parameter name
                    predicted_segments = self.diarizer_model.diarize(
                        audio=tmp_filename,
                        batch_size=1
                    )
                    
                    # Convert predicted segments to RTTM format
                    rttm_lines = self._segments_to_rttm(predicted_segments)
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_filename):
                        os.unlink(tmp_filename)

                output_tensor = pb_utils.Tensor(
                    "DIARIZATION_RTTM",
                    np.array(rttm_lines, dtype=object)
                )
                
                inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(inference_response)

            except Exception as e:
                import traceback
                self.logger.log_error(f"Error in diarizer: {str(e)}\n{traceback.format_exc()}")
                error = pb_utils.TritonError(message=f"Error in diarizer: {str(e)}")
                responses.append(pb_utils.InferenceResponse(output_tensors=[], error=error))
        
        return responses

    def _segments_to_rttm(self, predicted_segments):
        """
        Convert predicted segments to RTTM format.
        
        The diarize() method returns a list of tuples in format:
        (start_time, end_time, speaker_id)
        
        Args:
            predicted_segments: list of tuples (start, end, speaker_id)
        
        Returns:
            list of RTTM formatted strings
        """
        rttm_lines = []
        
        for segment in predicted_segments:
            start_time, end_time, speaker_id = segment
            duration = end_time - start_time
            
            # RTTM format: SPEAKER <file> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>
            rttm_line = f"SPEAKER audio 1 {start_time:.3f} {duration:.3f} <NA> <NA> speaker_{int(speaker_id)} <NA> <NA>"
            rttm_lines.append(rttm_line)
        
        return rttm_lines

    def finalize(self):
        self.logger.log_info("Diarizer model finalized.")
import math
from .common import remove_duplicates_and_blank

def class2str(target, char_dict):
    content = []
    for w in target:
        content.append(char_dict[int(w)])
    return ''.join(content).replace(' ',' ')

def milliseconds_to_hhmmssms(milliseconds):
    hours = milliseconds // (1000 * 60 * 60)
    remaining_ms = milliseconds % (1000 * 60 * 60)
    minutes = remaining_ms // (1000 * 60)
    remaining_ms %= (1000 * 60)
    seconds = remaining_ms // 1000
    remaining_ms %= 1000
    return f"{hours:02}:{minutes:02}:{seconds:02}:{remaining_ms:03}"

def get_output(hyps, char_dict):
    decodes = []
    for hyp in hyps:
        hyp = remove_duplicates_and_blank(hyp)
        decode = class2str(hyp, char_dict)
        decodes.append(decode)  
    return decodes

def get_word_timestamps(hyps, char_dict, time_scale=0.08):
    """
    Giải mã CTC và trả về danh sách các từ kèm timestamp.
    Fix lỗi gộp từ bằng cách định nghĩa chính xác ký tự U+2581.
    """
    batch_results = []
    
    # Scale thời gian: 80ms (0.08s) cho Subsampling 8
    FRAME_DURATION = 0.08 
    
    # [QUAN TRỌNG] Định nghĩa ký tự đặc biệt của SentencePiece bằng mã Unicode
    # U+2581: Lower One Eighth Block ( )
    SPIECE_UNDERLINE = '\u2581' 

    for tokens in hyps:
        tokens = tokens.cpu().numpy()
        words = []
        
        current_word_parts = []
        word_start_frame = -1
        
        for i, token in enumerate(tokens):
            if token == 0: # Blank token
                continue
            
            # Bỏ qua duplicate frames (đặc tính CTC)
            if i > 0 and token == tokens[i-1]:
                continue

            char = char_dict[int(token)]
            
            # --- LOGIC TÁCH TỪ ---
            # Kiểm tra xem token này có phải là bắt đầu từ mới không?
            # Các trường hợp:
            # 1. Bắt đầu bằng U+2581 (SentencePiece chuẩn)
            # 2. Bắt đầu bằng '_' (Một số tokenizer dùng gạch dưới)
            # 3. Bắt đầu bằng ' ' (Dấu cách thường)
            is_new_word_start = False
            if char.startswith(SPIECE_UNDERLINE) or char.startswith('_') or char.startswith(' '):
                is_new_word_start = True

            if is_new_word_start:
                # ĐÓNG TỪ CŨ (nếu có)
                if current_word_parts:
                    # Ghép các phần tử lại
                    full_word_raw = "".join(current_word_parts)
                    # Xóa các ký tự marker để lấy từ sạch
                    clean_word = full_word_raw.replace(SPIECE_UNDERLINE, "").replace("_", "").strip()
                    
                    if clean_word: # Chỉ lưu nếu từ có nội dung
                        words.append({
                            "word": clean_word,
                            "start": word_start_frame * FRAME_DURATION,
                            "end": i * FRAME_DURATION
                        })
                    
                    current_word_parts = []
                    word_start_frame = -1
                
                # BẮT ĐẦU TỪ MỚI
                word_start_frame = i
                current_word_parts.append(char)

            else:
                # TRƯỜNG HỢP NỐI TIẾP: Token là phần tiếp theo của từ (âm tiết phụ, vần...)
                if word_start_frame == -1:
                    word_start_frame = i # Trường hợp từ đầu tiên của câu không có marker
                current_word_parts.append(char)
        
        # LƯU TỪ CUỐI CÙNG (Flush)
        if current_word_parts:
             full_word_raw = "".join(current_word_parts)
             clean_word = full_word_raw.replace(SPIECE_UNDERLINE, "").replace("_", "").strip()
             if clean_word:
                words.append({
                    "word": clean_word,
                    "start": word_start_frame * FRAME_DURATION,
                    "end": len(tokens) * FRAME_DURATION
                })

        batch_results.append(words)

    return batch_results

# Giữ hàm cũ để tương thích
def get_output_with_timestamps(hyps, char_dict, max_silence=6):
    return []
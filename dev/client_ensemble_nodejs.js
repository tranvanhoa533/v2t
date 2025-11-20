#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');
const { ArgumentParser } = require('argparse');

// Đường dẫn đến file .proto của Triton (giữ nguyên cấu trúc thư mục dev/protos)
const PROTO_DIR = path.join(__dirname, 'protos');
const GRPC_SERVICE_PROTO_PATH = path.join(PROTO_DIR, 'grpc_service.proto');

// Load protobuf
let tritonGrpcService;
try {
    const packageDefinition = protoLoader.loadSync(
        GRPC_SERVICE_PROTO_PATH,
        {
            keepCase: true,
            longs: String,
            enums: String,
            defaults: true,
            oneofs: true,
            includeDirs: [PROTO_DIR]
        }
    );
    tritonGrpcService = grpc.loadPackageDefinition(packageDefinition).inference;
} catch (err) {
    console.error(`Failed to load protobuf definitions. Error: ${err.message}`);
    process.exit(1);
}

function main() {
    const parser = new ArgumentParser({
        description: 'Node.js gRPC client for Triton ASR + Diarization Ensemble.'
    });
    parser.add_argument('-u', '--url', {
        type: String,
        default: 'localhost:10001',
        help: 'Inference server URL. Default is localhost:10001.'
    });
    parser.add_argument('-m', '--model_name', {
        type: String,
        default: 'asr_diarization_ensemble', // Tên model mặc định cho ensemble
        help: 'Name of the model to use. Default is asr_diarization_ensemble.'
    });
    parser.add_argument('audio_file', {
        type: String,
        help: 'Path to the audio file (e.g., .wav, .mp3).'
    });
    parser.add_argument('--audio_format', {
        type: String,
        default: 'wav',
        help: "Format of the audio file (e.g., 'wav', 'mp3'). Default: 'wav'."
    });
    parser.add_argument('-v', '--verbose', {
        action: 'store_true',
        default: false,
        help: 'Enable verbose client output.'
    });

    const args = parser.parse_args();

    if (!fs.existsSync(args.audio_file)) {
        console.error(`Error: Audio file '${args.audio_file}' not found.`);
        process.exit(1);
    }

    // Cấu hình gRPC client
    const channelOptions = {
        'grpc.max_send_message_length': -1,
        'grpc.max_receive_message_length': -1,
    };

    const client = new tritonGrpcService.GRPCInferenceService(
        args.url,
        grpc.credentials.createInsecure(),
        channelOptions
    );

    // 1. Chuẩn bị Inputs
    const inputs = [];
    
    // Input 1: AUDIO_BYTES
    let audioBuffer;
    try {
        audioBuffer = fs.readFileSync(args.audio_file);
        inputs.push({
            name: 'AUDIO_BYTES',
            shape: [1],
            datatype: 'BYTES',
            contents: { bytes_contents: [audioBuffer] }
        });
    } catch (err) {
        console.error(`Error reading audio file: ${err.message}`);
        process.exit(1);
    }

    // Input 2: AUDIO_FORMAT (Tùy chọn nhưng được khuyến nghị)
    inputs.push({
        name: 'AUDIO_FORMAT',
        shape: [1],
        datatype: 'BYTES',
        contents: { bytes_contents: [Buffer.from(args.audio_format.toLowerCase(), 'utf8')] }
    });

    // 2. Chuẩn bị Output (Lưu ý tên output của Ensemble là FINAL_TRANSCRIPT)
    const outputs = [{ name: 'FINAL_TRANSCRIPT' }];

    // 3. Tạo Request
    const request = {
        model_name: args.model_name,
        model_version: '', // Lấy phiên bản mới nhất
        inputs: inputs,
        outputs: outputs
    };

    if (args.verbose) {
        console.log(`Client: Sending gRPC request to ${args.url} for model '${args.model_name}'...`);
        console.log(`Client: Audio file size: ${audioBuffer.length} bytes, Format: ${args.audio_format}`);
    }

    // 4. Gửi Request
    client.modelInfer(request, (err, response) => {
        if (err) {
            console.error('Client: gRPC Inference failed:', err.message);
            return;
        }

        // 5. Xử lý Output
        // Tìm tensor output có tên "FINAL_TRANSCRIPT"
        const finalTranscriptMeta = response.outputs.find(o => o.name === 'FINAL_TRANSCRIPT');
        
        if (!finalTranscriptMeta) {
            console.log('No FINAL_TRANSCRIPT output found in response.');
            return;
        }

        const outputIndex = response.outputs.indexOf(finalTranscriptMeta);
        const rawContent = response.raw_output_contents[outputIndex];

        if (rawContent && finalTranscriptMeta.shape && finalTranscriptMeta.shape.length > 0) {
            const numStrings = parseInt(finalTranscriptMeta.shape[0], 10);
            const results = parseBytesStringOutput(rawContent, numStrings);

            console.log('\n--- Final Transcript (ASR + Diarization) ---');
            results.forEach(line => console.log(line));
        } else {
            console.log('Received empty transcript.');
        }
    });
}

/**
 * Hàm helper để parse output dạng BYTES (TYPE_STRING) của Triton
 * Cấu trúc: [4-byte length][string bytes][4-byte length][string bytes]...
 */
function parseBytesStringOutput(buffer, numStrings) {
    const strings = [];
    let offset = 0;

    for (let i = 0; i < numStrings; i++) {
        if (offset + 4 > buffer.length) break;
        
        const len = buffer.readUInt32LE(offset);
        offset += 4;

        if (offset + len > buffer.length) break;

        const str = buffer.toString('utf8', offset, offset + len);
        strings.push(str);
        offset += len;
    }
    return strings;
}

if (require.main === module) {
    main();
}
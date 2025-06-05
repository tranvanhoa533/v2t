#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');
const { ArgumentParser } = require('argparse');

// Path to Triton's .proto files (adjust if your location is different)
const PROTO_DIR = path.join(__dirname, 'protos'); // Assumes a 'protos' subdirectory
const GRPC_SERVICE_PROTO_PATH = path.join(PROTO_DIR, 'grpc_service.proto');

// Load the protobuf
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
    console.error(`Failed to load protobuf definitions from ${GRPC_SERVICE_PROTO_PATH}. Ensure .proto files are in '${PROTO_DIR}'. Error: ${err.message}`);
    process.exit(1);
}


function main() {
    const parser = new ArgumentParser({
        description: 'Node.js gRPC client for Triton ChunkFormer model.'
    });
    parser.add_argument('-u', '--url', {
        type: String,
        default: 'localhost:10001',
        help: 'Inference server URL. Default is localhost:8001.'
    });
    parser.add_argument('-m', '--model_name', {
        type: String,
        default: 'v2t_vi', // Ensure this matches your model name in Triton
        help: 'Name of the model to use in Triton. Default is chunkformer.'
    });
    parser.add_argument('audio_file', {
        type: String,
        help: 'Path to the audio file to transcribe (e.g., .wav, .mp3).'
    });
    // Define client-side defaults that match server defaults if not provided
    parser.add_argument('--audio_format', {
        type: String,
        default: 'wav', // Client-side default, server also has a default
        help: "Format of the audio file (e.g., 'wav', 'mp3'). Default: 'wav'."
    });
    parser.add_argument('--total_batch_duration_sec', {
        type: 'int',
        default: 1800, // Client-side default
        help: 'Total audio duration (in seconds) in a batch. Default: 1800'
    });
    parser.add_argument('--chunk_size', {
        type: 'int',
        default: 64, // Client-side default
        help: 'Size of the chunks for processing. Default: 64'
    });
    parser.add_argument('--left_context_size', {
        type: 'int',
        default: 128, // Client-side default
        help: 'Size of the left context for attention. Default: 128'
    });
    parser.add_argument('--right_context_size', {
        type: 'int',
        default: 128, // Client-side default
        help: 'Size of the right context for attention. Default: 128'
    });
    parser.add_argument('--autocast_dtype_str', {
        type: String,
        default: 'NONE', // Client-side default
        choices: ["fp32", "bf16", "fp16", "NONE"],
        help: "Autocast dtype for server-side processing ('fp32', 'bf16', 'fp16', 'NONE'). Default: 'NONE'"
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
    if (tritonGrpcService === undefined || !fs.existsSync(GRPC_SERVICE_PROTO_PATH)) {
        console.error(`Error: Proto file '${GRPC_SERVICE_PROTO_PATH}' not found or failed to load. Please ensure Triton .proto files are in the 'protos' directory or update PROTO_DIR.`);
        process.exit(1);
    }

    const channelOptions = {
        'grpc.max_send_message_length': -1,
        'grpc.max_receive_message_length': -1,
    };

    const client = new tritonGrpcService.GRPCInferenceService(
        args.url,
        grpc.credentials.createInsecure(),
        channelOptions
    );

    const inputs = [];
    let audioBuffer;
    try {
        audioBuffer = fs.readFileSync(args.audio_file);
        inputs.push({
            name: 'AUDIO_BYTES',
            shape: [1],
            datatype: 'BYTES',
            contents: { bytes_contents: [audioBuffer] }
        });
        if (args.verbose) {
            console.log(`Client: Read audio file '${args.audio_file}' (${audioBuffer.length} bytes)`);
        }
    } catch (err) {
        console.error(`Error reading audio file: ${err.message}`);
        process.exit(1);
    }

    // Always send optional inputs, using user-provided value or client-side default
    inputs.push({
        name: 'AUDIO_FORMAT',
        shape: [1],
        datatype: 'BYTES',
        contents: { bytes_contents: [Buffer.from(args.audio_format.toLowerCase(), 'utf8')] }
    });
    if (args.verbose) console.log(`Client: Sending audio_format: ${args.audio_format.toLowerCase()}`);

    inputs.push({
        name: 'TOTAL_BATCH_DURATION_SEC',
        shape: [1],
        datatype: 'INT32',
        contents: { int_contents: [args.total_batch_duration_sec] }
    });
    if (args.verbose) console.log(`Client: Sending total_batch_duration_sec: ${args.total_batch_duration_sec}`);

    inputs.push({
        name: 'CHUNK_SIZE',
        shape: [1],
        datatype: 'INT32',
        contents: { int_contents: [args.chunk_size] }
    });
    if (args.verbose) console.log(`Client: Sending chunk_.size: ${args.chunk_size}`);

    inputs.push({
        name: 'LEFT_CONTEXT_SIZE',
        shape: [1],
        datatype: 'INT32',
        contents: { int_contents: [args.left_context_size] }
    });
    if (args.verbose) console.log(`Client: Sending left_context_size: ${args.left_context_size}`);
    
    inputs.push({
        name: 'RIGHT_CONTEXT_SIZE',
        shape: [1],
        datatype: 'INT32',
        contents: { int_contents: [args.right_context_size] }
    });
    if (args.verbose) console.log(`Client: Sending right_context_size: ${args.right_context_size}`);

    inputs.push({
        name: 'AUTOCAST_DTYPE_STR',
        shape: [1],
        datatype: 'BYTES',
        contents: { bytes_contents: [Buffer.from(args.autocast_dtype_str.toUpperCase(), 'utf8')] }
    });
    if (args.verbose) console.log(`Client: Sending autocast_dtype_str: ${args.autocast_dtype_str.toUpperCase()}`);


    const outputs = [{ name: 'TRANSCRIPTIONS' }];
    const request = {
        model_name: args.model_name,
        model_version: '', // Empty for latest
        inputs: inputs,
        outputs: outputs
    };

    if (args.verbose) {
        console.log(`Client: Sending gRPC request to Triton (${args.url}) for model '${args.model_name}'...`);
    
        // Create a deep copy specifically for logging if you need to modify nested properties
        const requestLog = JSON.parse(JSON.stringify(request)); // Simple deep copy, but loses Buffer types
    
        // Better: Build a log-specific object carefully
        const inputsForLog = request.inputs.map((input, index) => {
            const inputLog = { ...input }; // Shallow copy of the input object
            if (input.contents) {
                inputLog.contents = { ...input.contents }; // Shallow copy of contents
                if (input.name === 'AUDIO_BYTES' && input.contents.bytes_contents) {
                    // Assuming audioBuffer is accessible here or use input.contents.bytes_contents[0].length
                    inputLog.contents.bytes_contents = [`<${input.contents.bytes_contents[0].length} bytes of audio data>`];
                } else if (input.name === 'AUDIO_FORMAT' && input.contents.bytes_contents) {
                    inputLog.contents.bytes_contents = ["<format string bytes>"];
                } else if (input.name === 'AUTOCAST_DTYPE_STR' && input.contents.bytes_contents) {
                    inputLog.contents.bytes_contents = ["<autocast string bytes>"];
                }
                // Add other specific masking for other inputs if needed
            }
            return inputLog;
        });
    
        const finalRequestLogObject = {
            ...request, // Spread top-level properties from original request
            inputs: inputsForLog // Use the specially prepared inputs for logging
        };
    
        console.log("Client: Request object structure:", JSON.stringify(finalRequestLogObject, null, 2));
    }
    
    // The original 'request' object remains untouched and is sent to modelInfer

    client.modelInfer(request, (err, response) => {
        if (err) {
            console.error('Client: gRPC Inference failed:', err.message);
            if (err.details) console.error('Client: Details:', err.details);
            return;
        }
    
        if (args.verbose) {
            const responseLog = {...response};
            if (responseLog.raw_output_contents && responseLog.raw_output_contents.length > 0) {
                responseLog.raw_output_contents = responseLog.raw_output_contents.map(
                    b => `<${b.length} bytes of raw output data>`
                );
            }
            console.log('Client: Received response structure:', JSON.stringify(responseLog, null, 2));
        }
    
        const transcriptions = [];
        const transcriptionsOutputMetadata = response.outputs.find(o => o.name === 'TRANSCRIPTIONS');
        let outputIndex = -1;
    
        if (transcriptionsOutputMetadata) {
            outputIndex = response.outputs.indexOf(transcriptionsOutputMetadata);
        }
    
        if (outputIndex !== -1 &&
            response.raw_output_contents &&
            response.raw_output_contents.length > outputIndex &&
            transcriptionsOutputMetadata.shape &&
            transcriptionsOutputMetadata.shape.length > 0) {
    
            const rawBuffer = response.raw_output_contents[outputIndex]; // This is a Node.js Buffer
            const numStrings = parseInt(transcriptionsOutputMetadata.shape[0], 10); // Get the number of strings (e.g., 39)
    
            let offset = 0;
            for (let i = 0; i < numStrings; i++) {
                // Check if there's enough space for the length prefix
                if (offset + 4 > rawBuffer.length) {
                    console.error(`Client: Buffer exhausted when trying to read length for string ${i + 1}. Expected 4 bytes, got ${rawBuffer.length - offset}`);
                    break;
                }
    
                const stringLength = rawBuffer.readUInt32LE(offset); // Read 4-byte length for the current string
                offset += 4;
    
                // Check if there's enough space for the string itself
                if (offset + stringLength > rawBuffer.length) {
                    console.error(`Client: Buffer exhausted when trying to read string ${i + 1}. Expected ${stringLength} bytes, got ${rawBuffer.length - offset}`);
                    break;
                }
    
                const segment = rawBuffer.toString('utf8', offset, offset + stringLength);
                transcriptions.push(segment);
                offset += stringLength;
            }
    
            if (transcriptions.length > 0) {
                console.log('\nTranscription Results:');
                transcriptions.forEach(line => {
                    console.log(line);
                });
            } else if (numStrings > 0) {
                console.log('Error parsing transcription segments from raw buffer, though segments were expected.');
            } else {
                console.log('No transcription segments found, though output tensor was present.');
            }
    
        } else {
            console.log('No transcriptions received in raw_output_contents or output metadata/shape is unexpected.');
            if (args.verbose && response) {
                 if (response.outputs) console.log('Client: Full response outputs array:', response.outputs);
                 if (response.raw_output_contents) console.log('Client: Full response raw_output_contents:', response.raw_output_contents);
            } else if (args.verbose) {
                console.log('Client: Response object was null or did not contain expected output structures.');
            }
        }
    });
}

if (require.main === module) {
    main();
}

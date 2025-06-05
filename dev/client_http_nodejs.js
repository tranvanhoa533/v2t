#!/usr/bin/env node

const fs = require('fs');
const http = require('http'); // Using built-in http module
const path = require('path');
const { ArgumentParser } = require('argparse');
const { URL } = require('url'); // For parsing URL

function main() {
    const parser = new ArgumentParser({
        description: 'Node.js HTTP client for Triton ChunkFormer model.'
    });
    parser.add_argument('-u', '--url', {
        type: String,
        default: 'http://localhost:10000', // Default HTTP port and scheme
        help: 'Inference server URL. Default is http://localhost:8000.'
    });
    parser.add_argument('-m', '--model_name', {
        type: String,
        default: 'v2t_vi',
        help: 'Name of the model to use in Triton. Default is chunkformer.'
    });
    parser.add_argument('--model_version', {
        type: String,
        default: '', // Empty for latest version
        help: 'Version of the model to use. Default is latest.'
    });
    parser.add_argument('audio_file', {
        type: String,
        help: 'Path to the audio file to transcribe (e.g., .wav, .mp3).'
    });
    parser.add_argument('--audio_format', {
        type: String,
        default: 'wav',
        help: "Format of the audio file (e.g., 'wav', 'mp3'). Default: 'wav'."
    });
    parser.add_argument('--total_batch_duration_sec', {
        type: 'int',
        default: 1800,
        help: 'Total audio duration (in seconds) in a batch. Default: 1800'
    });
    parser.add_argument('--chunk_size', {
        type: 'int',
        default: 64,
        help: 'Size of the chunks for processing. Default: 64'
    });
    parser.add_argument('--left_context_size', {
        type: 'int',
        default: 128,
        help: 'Size of the left context for attention. Default: 128'
    });
    parser.add_argument('--right_context_size', {
        type: 'int',
        default: 128,
        help: 'Size of the right context for attention. Default: 128'
    });
    parser.add_argument('--autocast_dtype_str', {
        type: String,
        default: 'NONE',
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

    let audioBuffer;
    try {
        audioBuffer = fs.readFileSync(args.audio_file);
        if (args.verbose) {
            console.log(`Client: Read audio file '${args.audio_file}' (${audioBuffer.length} bytes)`);
        }
    } catch (err) {
        console.error(`Error reading audio file: ${err.message}`);
        process.exit(1);
    }

    // Construct inputs for the KServe v2 JSON payload
    const inputs = [];

    // AUDIO_BYTES: Needs to be base64 encoded
    inputs.push({
        name: 'AUDIO_BYTES',
        shape: [1], // Shape for a single item
        datatype: 'BYTES', // KServe v2 datatype for Triton's TYPE_STRING carrying bytes
        data: [audioBuffer.toString('base64')] // Data is an array of base64 encoded strings
    });

    // Optional inputs:
    // For string parameters that are TYPE_STRING in Triton's config.pbtxt,
    // the KServe JSON datatype is "BYTES", and the data is an array of the actual strings.
    if (args.audio_format) {
        inputs.push({
            name: 'AUDIO_FORMAT',
            shape: [1],
            datatype: 'BYTES',
            data: [args.audio_format.toLowerCase()]
        });
        if (args.verbose) console.log(`Client: Sending audio_format: ${args.audio_format.toLowerCase()}`);
    }

    if (args.total_batch_duration_sec) {
        inputs.push({
            name: 'TOTAL_BATCH_DURATION_SEC',
            shape: [1],
            datatype: 'INT32',
            data: [args.total_batch_duration_sec]
        });
        if (args.verbose) console.log(`Client: Sending total_batch_duration_sec: ${args.total_batch_duration_sec}`);
    }

    if (args.chunk_size) {
        inputs.push({
            name: 'CHUNK_SIZE',
            shape: [1],
            datatype: 'INT32',
            data: [args.chunk_size]
        });
        if (args.verbose) console.log(`Client: Sending chunk_size: ${args.chunk_size}`);
    }
    if (args.left_context_size) {
        inputs.push({
            name: 'LEFT_CONTEXT_SIZE',
            shape: [1],
            datatype: 'INT32',
            data: [args.left_context_size]
        });
         if (args.verbose) console.log(`Client: Sending left_context_size: ${args.left_context_size}`);
    }
    if (args.right_context_size) {
        inputs.push({
            name: 'RIGHT_CONTEXT_SIZE',
            shape: [1],
            datatype: 'INT32',
            data: [args.right_context_size]
        });
        if (args.verbose) console.log(`Client: Sending right_context_size: ${args.right_context_size}`);
    }
    if (args.autocast_dtype_str) {
        inputs.push({
            name: 'AUTOCAST_DTYPE_STR',
            shape: [1],
            datatype: 'BYTES', // For string parameters
            data: [args.autocast_dtype_str.toUpperCase()]
        });
        if (args.verbose) console.log(`Client: Sending autocast_dtype_str: ${args.autocast_dtype_str.toUpperCase()}`);
    }


    // Construct the full request payload
    const requestPayload = {
        model_name: args.model_name,
        model_version: args.model_version, // Can be empty for latest
        inputs: inputs,
        outputs: [
            { name: 'TRANSCRIPTIONS' }
        ]
    };

    const jsonData = JSON.stringify(requestPayload);

    // Parse URL to get hostname, port, and path
    const parsedUrl = new URL(args.url);
    const modelInferPath = `/v2/models/${args.model_name}${args.model_version ? '/versions/' + args.model_version : ''}/infer`;

    const options = {
        hostname: parsedUrl.hostname,
        port: parsedUrl.port || (parsedUrl.protocol === 'https:' ? 443 : 80),
        path: modelInferPath,
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(jsonData),
            // 'Inference-Header-Content-Length': Buffer.byteLength(jsonData) // May be needed by some Triton versions/configs
        }
    };

    if (args.verbose) {
        console.log(`Client: Sending HTTP POST request to ${parsedUrl.protocol}//${options.hostname}:${options.port}${options.path}`);
        console.log("Client: Request payload:", jsonData);
    }

    const req = http.request(options, (res) => {
        if (args.verbose) {
            console.log(`Client: STATUS: ${res.statusCode}`);
            console.log('Client: HEADERS:', JSON.stringify(res.headers, null, 2));
        }
        let responseBody = '';
        res.setEncoding('utf8');
        res.on('data', (chunk) => {
            responseBody += chunk;
        });
        res.on('end', () => {
            if (res.statusCode === 200) {
                try {
                    const responseJson = JSON.parse(responseBody);
                    if (args.verbose) {
                        console.log('Client: Received response body (parsed JSON):', JSON.stringify(responseJson, null, 2));
                    }

                    const transcriptionsOutput = responseJson.outputs.find(o => o.name === 'TRANSCRIPTIONS');

                    if (transcriptionsOutput && transcriptionsOutput.data && transcriptionsOutput.data.length > 0) {
                        console.log('\nTranscription Results:');
                        transcriptionsOutput.data.forEach(line => {
                            // For "BYTES" datatype carrying strings, data is an array of strings
                            // If it was binary data that got base64 decoded on server and sent as string,
                            // you might need Buffer.from(line, 'base64').toString('utf8')
                            // But here, server's python backend returns plain strings in the list for TRANSCRIPTIONS.
                            console.log(line);
                        });
                    } else {
                        console.log('No transcriptions received or output format is unexpected.');
                         if (transcriptionsOutput && transcriptionsOutput.data && transcriptionsOutput.data.length === 0) {
                            console.log('Server returned an empty list of transcriptions.');
                        }
                    }
                } catch (e) {
                    console.error('Client: Failed to parse JSON response:', e.message);
                    if (args.verbose) console.error('Client: Raw response body:', responseBody);
                }
            } else {
                console.error(`Client: Request failed. Status Code: ${res.statusCode}`);
                if (args.verbose) console.error('Client: Raw response body:', responseBody);
            }
        });
    });

    req.on('error', (e) => {
        console.error(`Client: Problem with request: ${e.message}`);
    });

    // Write data to request body
    req.write(jsonData);
    req.end();
}

if (require.main === module) {
    main();
}

module.exports = [
  {
    slug: "indus-web",
    title: "Official Website Of Indus Real Estate LLC (Dubai)",
    overview:
      "Indus main website, powered by Next.js, PHP, and Node.js, delivers top-notch performance, SEO, and user engagement. With Next.js' image optimization and PHP/Node.js backend, we achieve fast load times and dynamic experiences. Prioritizing speed and scalability, we optimize all elements, from image compression to server-side rendering. With clean code and responsive layouts, our site boosts SEO, driving visibility and ranking. Continuously monitored and optimized, our site ensures an engaging user experience, spurring conversions in the real estate market.",
    links: {
      website: "www.indusre.com",
      github: "indus_new_web",
    },
    images: [
      "indus_web/indusre.com_1.png",
      "indus_web/indusre.com_2.png",
      "indus_web/indusre.com_3.png",
      "indus_web/indusre.com_4.png",
      "indus_web/indusre.com_5.png",
      "indus_web/indusre.com_6.png",
      "indus_web/indusre.com_7.png",
    ],
    code: [],
    mobile: false,
    description: ``,
    features: ["", "", ""],
    built_with: [
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
    ],
    published: true,
    date: "2023-09-16",
  },
  {
    slug: "indus-premium",
    title: "Indus Website for premium properties.",
    overview:
      "Introducing exclusive platform for premium real estate, powered by Next.js, PHP, and Node.js. Accessible only to registered users, unique UI/UX prioritizes luxury and privacy over SEO. With lightning-fast performance and dynamic functionality, it deliver elite browsing experiences for discerning clientele.",
    links: {
      website: "indus-premium.vercel.app",
      github: "melanie_new_web",
    },
    images: [
      "indus_premium/indus-premium.vercel.app_1.png",
      "indus_premium/indus-premium.vercel.app_2.png",
      "indus_premium/indus-premium.vercel.app_3.png",
      "indus_premium/indus-premium.vercel.app_4.png",
      "indus_premium/indus-premium.vercel.app_5.png",
      "indus_premium/indus-premium.vercel.app_6.png",
      "indus_premium/indus-premium.vercel.app_7.png",
      "indus_premium/indus-premium.vercel.app_8.png",
      "indus_premium/indus-premium.vercel.app_9.png",
      "indus_premium/indus-premium.vercel.app_10.png",
      "indus_premium/indus-premium.vercel.app_11.png",
    ],
    code: [],
    mobile: false,
    description: "",
    features: ["", "", ""],
    built_with: [
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
    ],
    published: true,
    date: "2023-11-03",
  },
  {
    slug: "indus-cms",
    title: "Content Management System for Website (Angular)",
    overview:
      "Introducing the CMS system, crafted with Angular for the frontend and backed by PHP, Node.js, and Python for the backend. With a sleek UI/UX powered by Angular Material Library, it optimizes workflow efficiency. From content management to optimization tools, every feature is designed for seamless integration and enhanced productivity.",
    links: {
      website: "",
      github: "indus-cms",
    },
    images: [
      "indus_cms/indusre.app_1.png",
      "indus_cms/indusre.app_2.png",
      "indus_cms/indusre.app_3.png",
      "indus_cms/indusre.app_4.png",
      "indus_cms/indusre.app_5.png",
      "indus_cms/indusre.app_6.png",
      "indus_cms/indusre.app_7.png",
      "indus_cms/indusre.app_8.png",
      "indus_cms/indusre.app_9.png",
      "indus_cms/indusre.app_10.png",
      "indus_cms/indusre.app_11.png",
      "indus_cms/indusre.app_12.png",
      "indus_cms/indusre.app_13.png",
    ],
    code: [],
    mobile: false,
    description: "",
    features: ["", "", ""],
    built_with: [
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
    ],
    published: true,
    date: "2023-12-01",
  },
  {
    slug: "chat-bot",
    title: "TensorFlow-Powered Chatbot Project (LLM)",
    overview:
      "Introducing my real estate chatbot project, powered by Python, NumPy, and TensorFlow's Sequential model. By utilizing Pad sequences, this innovative solution transforms raw data into meaningful conversational interactions. From seamless query responses to personalized recommendations, this chatbot redefines real estate engagement with unparalleled efficiency and accuracy.",
    links: {
      website: "",
      github: "chat_bot_llm_py",
    },
    images: [],
    code: [
      {
        title: "Custom Language Model Training with TensorFlow",
        description:
          "This Python code implements a custom language model using TensorFlow, allowing you to train and generate text sequences. By tokenizing text data and preparing input sequences, the code defines a Sequential model architecture with LSTM layers. After compiling the model, it is trained on the input sequences to learn patterns in the text data. Finally, the trained model is saved for future use, enabling text generation based on the learned patterns.",
        features: [],
        code: `import numpy as np\nimport tensorflow as tf
        \nTokenizer = tf.keras.preprocessing.text.Tokenizer\npad_sequences = tf.keras.preprocessing.sequence.pad_sequences
        \nSequential = tf.keras.models.Sequential\nEmbedding = tf.keras.layers.Embedding\nSimpleRNN = tf.keras.layers.SimpleRNN\nDense = tf.keras.layers.Dense\nLSTM = tf.keras.layers.LSTM\nDropout = tf.keras.layers.Dropout
        \n# Load your text data\n# Here I'm simply loading a relative file which contains the array of data (data.py)\nfrom data import text_data_arr
        \n# Tokenize the text\ntokenizer = Tokenizer(char_level=True, lower=True)\ntokenizer.fit_on_texts(text_data_arr)
        \n# Convert text to sequences\nsequences = tokenizer.texts_to_sequences(text_data_arr)[0]
        \n# Prepare input and target sequences\ninput_sequences = []\noutput_sequences = []
        \nsequence_length = 100\nfor i in range(len(sequences) - sequence_length):\n    input_sequences.append(sequences[i:i + sequence_length])\n    output_sequences.append(sequences[i + sequence_length])
        \ninput_sequences = np.array(input_sequences)\noutput_sequences = np.array(output_sequences)
        \nvocab_size = len(tokenizer.word_index) + 1
        \n# Define the model architecture:\nmodel = Sequential([
        # Embedding layer that maps each word in the input sequence to a dense vector
        Embedding(vocab_size, 32, input_length=sequence_length),
        # First LSTM layer with 128 units, returning a sequence of outputs for each time step
        LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        # Second LSTM layer with 128 units, returning only the final output for the whole sequence
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        # Dense layer with a softmax activation, outputting a probability distribution over the vocabulary
        Dense(vocab_size, activation="softmax"),\n])
        \nmodel.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])\nmodel.summary()
        \n# Train the model\nepochs = 250  # Increase the number of epochs to give the model more time to learn\nbatch_size = 32\nmodel.fit(input_sequences, output_sequences, epochs=epochs, batch_size=batch_size)
        \nmodel.save('custom_llm_model.keras')`,
        lang: "python",
      },
      {
        title:
          "Predictive Text Generation with Custom Language Models using TensorFlow",
        description:
          "This Python code showcases how to utilize a custom language model trained with TensorFlow for text generation. The script loads a pre-trained custom language model, tokenizes input text, pads sequences to a fixed length, and feeds them into the model for prediction. Specifically, it demonstrates the process of predicting the next token in a sequence using an LSTM layer. The code offers insights into reshaping input data for LSTM compatibility and obtaining predictions from the model's output layer. This can serve as a foundational guide for integrating and testing custom language models within TensorFlow applications.",
        features: [],
        code: `import tensorflow as tf\nfrom tensorflow.keras.preprocessing.sequence import pad_sequences\nimport numpy as np
        \n# Load your custom language model\ncustom_model = tf.keras.models.load_model('custom_llm_model.h5')
        \n# Generate text samples\ninput_text = "What is real estate in dubai?"\ntokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")\ntokenizer.fit_on_texts([input_text])\ninput_ids = tokenizer.texts_to_sequences([input_text])\ninput_ids_padded = pad_sequences(input_ids, maxlen=50, padding='post', truncating='post')
        \n# Ensure input sequence has length 50 by padding\ninput_ids_padded = pad_sequences(input_ids, maxlen=50, padding='post', truncating='post')
        \n# Print shapes for debugging\nprint("Input shape before prediction:", input_ids_padded.shape)
        \n# Reshape input for the LSTM layer\ninput_ids_3d = tf.expand_dims(input_ids_padded, axis=-1)
        \n# Define a function for predicting the next token using the LSTM layer\n@tf.function\ndef lstm_predict_step(input_data):\n    return custom_model.get_layer('lstm')(input_data)
        \n# Apply the function to get the LSTM output\nlstm_output = lstm_predict_step(input_ids_3d)
        \n# Continue with the rest of the prediction\noutput_ids = custom_model.get_layer('dense')(lstm_output)
        \n# Print shapes for debugging\nprint("Output shape after prediction:", output_ids.shape)`,
        lang: "python",
      },
      {
        title:
          "Text Generation Model Training from PDFs using TensorFlow and PyPDF2",
        description:
          "This Python script demonstrates the process of training a text generation model using PDF documents as a corpus. It begins by extracting text from a PDF file using PyPDF2, tokenizing the text, and converting it into sequences for model training. The script then pads the sequences to ensure uniform length, defines hyperparameters for the LSTM-based model, and compiles it with appropriate loss and optimizer settings. After splitting the data into training and validation sets, the model is trained and evaluated. Finally, the trained model is saved for future use. This code serves as a practical guide for building text generation models from PDF sources using TensorFlow and PyPDF2 libraries.",
        features: [],
        code: `from PyPDF2 import PdfReader\nfrom tensorflow.keras.preprocessing.text import Tokenizer\nfrom tensorflow.keras.preprocessing.sequence import pad_sequences\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import Embedding, LSTM, Dense\nfrom sklearn.model_selection import train_test_split\nimport numpy as np
        \n# Function to extract text from PDF using PyMuPDF\ndef extract_text_from_pdf(pdf_path):\n    reader = PdfReader(pdf_path)\n    text = ''\n    for page in reader.pages:\n        text += page.extract_text()\n    return text
        \n# Example PDF file path\npdf_path = 'assets/book.pdf'
        \n# Extract text from PDF\npdf_text = extract_text_from_pdf(pdf_path)
        \n# Tokenize the text\ntokenizer = Tokenizer(oov_token="<OOV>")\ntokenizer.fit_on_texts([pdf_text])
        \n# Convert text to sequences\nsequences = tokenizer.texts_to_sequences([pdf_text])
        \ninput_sequences = []\noutput_sequences = []
        \nfor sequence in sequences:\n    for i in range(1, len(sequence)):\n        input_sequences.append(sequence[:i])\n        output_sequences.append(sequence[i])
        \n# Pad sequences\nmax_seq_length = 50  # Adjust as needed\npadded_sequences = pad_sequences(input_sequences, maxlen=max_seq_length)
        \n# Convert to NumPy arrays\nX = np.array(padded_sequences)\ny = np.array(output_sequences)
        \n# Define hyperparameters\nvocab_size = len(tokenizer.word_index) + 1\nembedding_dim = 128\nlstm_units = 256\noutput_units = vocab_size
        \n# Build the model\nmodel = Sequential([\n   Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length),\n   LSTM(units=lstm_units),\n   Dense(units=output_units, activation='softmax')\n])
        \n# Compile the model\nmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        \n# Split the data into training and validation sets\nX_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        \n# Train the model\nmodel.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
        \n# Display model summary\nmodel.summary()\nmodel.save('custom_llm_model.h5')`,
        lang: "python",
      },
    ],
    mobile: false,
    description: "",
    features: ["", "", ""],
    built_with: [
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
    ],
    published: true,
    date: "2024-02-23",
  },
  {
    slug: "event-management-app",
    title: "Event Management App (Flutter)",
    overview:
      "Flutter-based event management app, developed with expertise in cross-platform optimization. Streamlining event planning and attendance tracking, this app integrates QR scanning and backend functionalities in PHP, Node.js, and SQL. With user-friendly interfaces for iOS and Android, it ensures seamless navigation and widespread adoption, revolutionizing event management processes.",
    links: {
      website: "",
      github: "qr_code_read_app",
    },
    images: [
      "e_mng/e_mng_1.webp",
      "e_mng/e_mng_2.webp",
      "e_mng/e_mng_3.webp",
      "e_mng/e_mng_4.webp",
    ],
    code: [],
    mobile: true,
    description: "",
    features: ["", "", ""],
    built_with: [
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
    ],
    published: true,
    date: "2024-03-15",
  },
  {
    slug: "property-management",
    title: "Property Management Cloud App (Angular)",
    overview:
      "Introducing Property Management Cloud PWA, a cutting-edge solution for Dubai real estate. Built with Angular, PHP, Node.js, and SQL, the platform offers seamless property, unit, and lease contract management. Leveraging Firebase for push notifications and featuring a three-tier authentication system, the solution revolutionizes real estate operations with efficiency and security. With a dynamic dashboard and custom styling using SCSS, it provides intuitive navigation and a visually appealing user experience.",
    links: {
      website: "",
      github: "IndusRealEstate_Customer_Portal",
    },
    images: ["pr_mng/pr_mng_1.jpg"],
    code: [],
    mobile: false,
    description:
      "This is an old project, Due to some internal issues the project got shutdown. Please checkout the github for the code.",
    features: ["", "", ""],
    built_with: [
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
    ],
    published: true,
    date: "2022-12-27",
  },
  {
    slug: "image-optimization",
    title: "Sharp Image Optimization (Node.js)",
    overview:
      "Node.js backend application for image optimization, a powerful tool built with Sharp, Express, Multer, and Compression. Designed to enhance performance and reduce file sizes, this app optimizes images based on input parameters sent via API. From image dimensions and quality to export type and compression size, it ensures efficient optimization tailored to specific requirements. With seamless integration and robust functionality, it streamlines the image optimization process, empowering developers to enhance performance and user experience effortlessly.",
    links: {
      website: "",
      github: "sharp_nodejs",
    },
    images: [],
    code: [
      {
        title:
          "Image Processing APIs with Express.js and Sharp for Bulk Resize and Optimization",
        description:
          "This Node.js code sets up an Express.js server to provide APIs for bulk image resizing and optimization using the Sharp library. The server utilizes CORS for cross-origin requests and Multer for handling file uploads. It defines two main APIs:",
        features: [
          {
            title: "Bulk Resize API",
            desc: "This API (/bulk-resize) accepts an array of image files and resize options as input. It iterates over the uploaded images, resizes them based on the provided options (format, scale, quality), and returns a zip file containing the resized images.",
          },
          {
            title: "Optimize Image API",
            desc: "The second API (/optimize-image) is designed to optimize a single image. It receives image data and optimization options, then processes the image accordingly (resizing, format conversion, quality adjustment), and returns the optimized image data as base64 encoded string.",
          },
        ],
        code: `const express = require("express");\nconst cors = require("cors");\nconst sharp = require("sharp");\nconst fs = require("fs");\nconst archiver = require("archiver");\nconst app = express();\nconst port = 3000;
        \nconst multer = require("multer");\nconst storage = multer.memoryStorage(); // Store files in memory
        \nconst upload = multer({\n   storage: storage,\n   limits: { fileSize: 1000 * 1024 * 1024 },\n});
        \n// Enable CORS\napp.use(cors());
        \napp.use(express.json({ limit: "1000mb" }));\napp.use(express.urlencoded({ limit: "1000mb", extended: true }));
        \n// Initialize compression module\nconst compression = require('compression'); 
        \n// Compress all HTTP responses\napp.use(compression()); 
        \nconst myConsole = new console.Console(fs.createWriteStream("./error.txt"));
        \n////////////////// ----------------- BULK RESIZE API ------------------- /////////////////////
        \napp.post("/bulk-resize", upload.array("files[]"), async (req, res) => {
        const files = req.files;
        
        const options = JSON.parse(req.body.options);
        let optimizedImageBuffer_arr = [];
        
          try {
            var promises = files.map(async function (image, index) {
              const img_info = JSON.parse(req.body['img_name_and_index']);
        
              const imageBuffer = Buffer.from(image.buffer);
        
              const new_width = Math.round(
                Math.min(img_info.width * (options.scale / 100))
              );
              const new_height = Math.round(
                Math.min((new_width * img_info.height) / img_info.width)
              );
        
              switch (options.format) {
                case "jpg":
                  await sharp(imageBuffer)
                    .resize(new_width, new_height)
                    .jpeg({
                      quality: options.quality,
                    })
                    .toBuffer()
                    .then((data) => {
                      optimizedImageBuffer_arr.push(data);
                    })
                    .catch(function (err) {
                      myConsole.error(err);
                    });
                  break;
                case "png":
                  await sharp(imageBuffer)
                    .resize(new_width, new_height)
                    .png({ quality: options.quality, effort: 1 })
                    .toBuffer()
                    .then((data) => {
                      optimizedImageBuffer_arr.push(data);
                    })
                    .catch(function (err) {
                      myConsole.error(err);
                    });
                  break;
                case "webp":
                  await sharp(imageBuffer)
                    .resize(new_width, new_height)
                    .webp({
                      quality: options.quality,
                      alphaQuality: options.quality,
                      effort: 0,
                    })
                    .toBuffer()
                    .then((data) => {
                      optimizedImageBuffer_arr.push(data);
                    })
                    .catch(function (err) {
                      myConsole.error(err);
                    });
                  break;
                case "avif":
                  await sharp(imageBuffer)
                    .resize(new_width, new_height)
                    .avif({ quality: options.quality, effort: 1 })
                    .toBuffer()
                    .then((data) => {
                      optimizedImageBuffer_arr.push(data);
                    })
                    .catch(function (err) {
                      myConsole.error(err);
                    });
                  break;
                case "tiff":
                  await sharp(imageBuffer)
                    .resize(new_width, new_height)
                    .tiff({
                      quality: options.quality,
                      compression: "lzw",
                      bitdepth: 1,
                    })
                    .toBuffer()
                    .then((data) => {
                      optimizedImageBuffer_arr.push(data);
                    })
                    .catch(function (err) {
                      myConsole.error(err);
                    });
                  break;
                default:
                  await sharp(imageBuffer)
                    .resize(new_width, new_height)
                    .jpeg({ quality: options.quality })
                    .toBuffer()
                    .then((data) => {
                      optimizedImageBuffer_arr.push(data);
                    })
                    .catch(function (err) {
                      myConsole.error(err);
                    });
                  break;
              }
            });
        
            Promise.all(promises).then(function () {
              let count = 0;
              // let final_base64_str = "";
              fs.mkdir("temp", (error) => {
                if (error) {
                  myConsole.log(error);
                } else {
                  optimizedImageBuffer_arr.forEach(async (chunk, i) => {
                    fs.appendFile(
                      'temp/index_img.fromat',
                      Buffer.from(chunk),
                      function (err) {
                        if (err) {
                          myConsole.log(err);
                        } else {
                          // console.log(chunk.length);
                        }
                      }
                    );
                    count++;
        
                    if (count == optimizedImageBuffer_arr.length) {
                      await zipDirectory("temp/", "target.zip");
                      const final_base64_str = fs.readFileSync("target.zip", {
                        encoding: "base64",
                      });
        
                      res.setHeader("Content-Type", "application/json");
                      res.send({ data: final_base64_str });
        
                      fs.rmSync("temp/", { recursive: true, force: true });
                      fs.unlinkSync("target.zip");
                    }
                  });
                }
              });
            });
          } catch (error) {
            myConsole.log(error);
          }\n});
        \nfunction zipDirectory(sourceDir, outPath) {
          const archive = archiver("zip", { zlib: { level: 9 } });
          const stream = fs.createWriteStream(outPath);
        
          return new Promise((resolve, reject) => {
            archive
              .directory(sourceDir, false)
              .on("error", (err) => reject(err))
              .pipe(stream);
        
            stream.on("close", () => resolve());
            archive.finalize();
          });\n}
        \n////////////// ----------------- OPTIMIZE IMAGE API ------------------- //////////////////////////
        \napp.post("/optimize-image", async (req, res) => {
          try {
            const { imageData, options } = req.body;
        
            const imageBuffer = Buffer.from(imageData, "base64");
        
            let optimizedImageBuffer;
        
            switch (options.format) {
              case "jpg":
                optimizedImageBuffer = await sharp(imageBuffer)
                  .resize(options.width, options.height)
                  .jpeg({
                    quality: options.quality,
                  })
                  .toBuffer();
                break;
              case "png":
                optimizedImageBuffer = await sharp(imageBuffer)
                  .resize(options.width, options.height)
                  .png({ quality: options.quality, effort: 1 })
                  .toBuffer();
              case "webp":
                optimizedImageBuffer = await sharp(imageBuffer)
                  .resize(options.width, options.height)
                  .webp({
                    quality: options.quality,
                    alphaQuality: options.quality,
                    effort: 0,
                    smartSubsample: true,
                  })
                  .sharpen()
                  .toBuffer();
                break;
              case "avif":
                optimizedImageBuffer = await sharp(imageBuffer)
                  .resize(options.width, options.height)
                  .avif({ quality: options.quality, effort: 0 })
                  .toBuffer();
                break;
              case "tiff":
                optimizedImageBuffer = await sharp(imageBuffer)
                  .resize(options.width, options.height)
                  .tiff({ quality: options.quality, compression: "lzw", bitdepth: 1 })
                  .toBuffer();
                break;
        
              default:
                optimizedImageBuffer = await sharp(imageBuffer)
                  .resize(options.width, options.height)
                  .jpeg({ quality: options.quality })
                  .toBuffer();
                break;
            }
        
            // Set appropriate response headers
            res.setHeader("Content-Type", "application/json");
            // res.setHeader("Content-Length", optimizedImageBuffer.byteLength);
        
            res.send({ data: optimizedImageBuffer.toString("base64") });
          } catch (error) {
            console.error("Error optimizing image:", error);
            res
              .status(500)
              .json({ error: "Internal Server Error", details: error.message });
          }\n});
        \nconst server = app.listen(port, () => {\n   console.log('Server is running at http://localhost:port');\n});
        \nserver.timeout = 6000000;`,
        lang: "javascript",
      },
    ],
    mobile: false,
    description: "",
    features: ["", "", ""],
    built_with: [
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
    ],
    published: true,
    date: "2023-08-20",
  },
  {
    slug: "image-upscale",
    title: "Image Upscale Ai (TensorFlow)",
    overview:
      "Python project for image upscaling, leveraging TensorFlow and Flask. This application allows users to upscale images to super resolution using the ESRGAN model. With Flask handling API requests, users can input images for upscaling, which are then processed using TensorFlow. The ESRGAN model, loaded from TensorFlow Hub, enhances image quality and detail, producing high-resolution results. The upscaled images are then returned to the user via the API, ready for further use. This project showcases the power of machine learning in image processing and provides a practical solution for enhancing image quality in real-world applications.",
    links: {
      website: "",
      github: "upscale_img_py",
    },
    images: [],
    code: [
      {
        title: "Image Upscaling Service using TensorFlow and Flask",
        description: `This Python script creates a web service for upscaling images using TensorFlow and Flask. It sets up a Flask server that exposes two endpoints: /img for processing images and /date for retrieving the server's current date.

        The /img endpoint accepts a POST request containing a base64 encoded image. It then decodes the image, preprocesses it to make it ready for the upscaling model, and utilizes a pre-trained ESRGAN (Enhanced Super-Resolution Generative Adversarial Network) model loaded from TensorFlow Hub to upscale the image. After upscaling, it converts the image back to base64 encoding and returns the resulting super-resolution image.
        
        The script utilizes libraries like TensorFlow Hub, PIL (Python Imaging Library), and Flask for image processing, handling HTTP requests, and serving the model predictions. This image upscaling service can be deployed to provide high-quality image upscaling capabilities in web applications.`,
        features: [],
        code: `from flask import Flask, jsonify, request\nfrom flask_cors import CORS\nimport subprocess\nimport os\nimport time\nfrom PIL import Image\nimport numpy as np\nimport tensorflow as tf\nimport tensorflow_hub as hub\nimport matplotlib.pyplot as plt\nimport base64\nimport io\nfrom io import BytesIO\nos.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
        \napp = Flask(__name__)\nCORS(app)
        \nSAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
        \ndef preprocess_image(image_data):\n    """ Loads image from path and preprocesses to make it model ready\n        Args:\n          image_path: Path to the image file\n    """
        \n    hr_image = tf.image.decode_image(image_data)\n    # If PNG, remove the alpha channel. The model only supports\n    # images with 3 color channels.\n    if hr_image.shape[-1] == 4:\n       hr_image = hr_image[...,:-1]
        \n    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4\n    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])\n    hr_image = tf.cast(hr_image, tf.float32)\n    return tf.expand_dims(hr_image, 0)
        \ndef save_image(image, filename):\n    """\n     Saves unscaled Tensor Images.\n     Args:\n       image: 3D image tensor. [height, width, channels]\n       filename: Name of the file to save.\n    """
        \n    if not isinstance(image, Image.Image):\n       image = tf.clip_by_value(image, 0, 255)\n       image = Image.fromarray(tf.cast(image, tf.uint8).numpy())\n   #   image.save("%s.jpg" % filename)\n   #   print("Saved as %s.jpg" % filename)
        \n    return image
        \ndef plot_image(image, title=""):\n    """ Plots images from image tensors.\n        Args:\n          image: 3D image tensor. [height, width, channels].\n          title: Title to display in the plot.\n    """
        \n    image = np.asarray(image)\n    image = tf.clip_by_value(image, 0, 255)\n    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())\n    plt.imshow(image)\n    plt.axis("off")\n    plt.title(title)
        \n@app.route('/img', methods=['POST'])\ndef get_img():\n    img = request.json['img']\n    img_d = base64.b64decode(img)\n    img_w = base64.urlsafe_b64encode(img_d)
        \n    img_b = tf.io.decode_base64(img_w)\n    hr_image = preprocess_image(img_b)
        \n    # Plotting Original Resolution image\n    # plot_image(tf.squeeze(hr_image), title="Original Image")\n    # save_image(tf.squeeze(hr_image), filename="Original Image")
        \n    model = hub.load(SAVED_MODEL_PATH)
        \n    fake_image = model(hr_image)\n    fake_image = tf.squeeze(fake_image)
        \n    # Plotting Super Resolution Image\n    plot_image(tf.squeeze(fake_image), title="Super Resolution")\n    s_img = save_image(tf.squeeze(fake_image), filename="Super Resolution")
        \n    buffered = BytesIO()\n    s_img.save(buffered, format="PNG")\n    img_str = base64.b64encode(buffered.getvalue())
        \n    # print(img_str)\n    return jsonify({'img': 'success', 'data' : img_str.decode('ascii')})
        \n@app.route('/date', methods=['GET'])\ndef get_date():\n    return jsonify({'date': '2020-01-01'})
        \nif __name__ == '__main__':\n    app.run()`,
        lang: "python",
      },
    ],
    mobile: false,
    description: "",
    features: ["", "", ""],
    built_with: [
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
    ],
    published: true,
    date: "2024-04-02",
  },
  ////////////////////////////////
  {
    slug: "object-detection",
    title: "Real-Time Object Detection (YOLO)",
    overview:
      "This project showcases a real-time object detection web application built using Python, Flask, Socket.IO, and OpenCV. Leveraging the YOLO (You Only Look Once) deep learning framework, the application can detect objects within images sent from a client interface in real-time. The backend, developed in Python, processes the images, detects objects, and sends the results back to the client via Socket.IO. The client interface, implemented using Next.js and Firebase for authentication, provides a seamless user experience for interacting with the object detection functionality. This project demonstrates the integration of machine learning models into web applications for real-time processing and interaction.",
    links: {
      website: "",
      github: "object_detector_py",
    },
    images: [],
    code: [
      {
        title: "Backend Python Usage Overview: Real-Time Object Detection",
        description: `Utilizing Flask and Flask-SocketIO, this Python backend integrates a YOLO (You Only Look Once) deep learning model with OpenCV for real-time object detection. Upon receiving base64-encoded image data via WebSocket connections, the backend decodes, processes, and analyzes the images. Detected objects, complete with bounding box coordinates and confidence scores, are promptly communicated back to the client interface via Socket.IO for instantaneous visualization. Robust error handling ensures smooth operation, while performance optimizations, including image resizing and asynchronous processing, enhance scalability and responsiveness.`,
        features: [],
        code: `from flask import Flask, jsonify, request\nfrom flask_socketio import SocketIO, emit\nimport cv2\nimport numpy as np
        \n# from flask_cors import CORS\nimport base64\nfrom PIL import Image\nimport io\nimport json
        \napp = Flask(__name__)\napp.config["SECRET_KEY"] = "secret"\nsocketio = SocketIO(app)\nsocketio.init_app(app, cors_allowed_origins="*")\n# CORS(app, origins="http://localhost:4200")
        \n# Load YOLO\nnet = cv2.dnn.readNet("assets/yolov3.weights", "assets/yolov3.cfg")\nclasses = []\nwith open("assets/coco.names", "r") as f:\n     classes = [line.strip() for line in f.readlines()]\nlayer_names = net.getLayerNames()\noutput_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        \n# Function to perform object detection on an image\ndef detect_objects(image):\n# Resize and normalize image\n    img = cv2.resize(image, None, fx=0.4, fy=0.4)\n    height, width, channels = img.shape
        \n    # Detect objects\n    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)\n    net.setInput(blob)\n    outs = net.forward(output_layers)
        \n    # Process detections\n    class_names_detected = []\n    confidences = []\n    boxes = []
        \n    for out in outs:\n        for detection in out:\n            scores = detection[5:]\n            class_id = np.argmax(scores)\n            confidence = scores[class_id]
        \n            if confidence > 0.5:\n                # Object detected\n                center_x = int(detection[0] * width)\n                center_y = int(detection[1] * height)\n                w = int(detection[2] * width)\n                h = int(detection[3] * height)
        \n                # Rectangle coordinates\n                x = int(center_x - w / 2)\n                y = int(center_y - h / 2)
        \n                boxes.append([x, y, w, h])\n                confidences.append(float(confidence))
        \n                if classes[class_id] not in class_names_detected:\n                   class_names_detected.append(classes[class_id])
        \n    return boxes, confidences, class_names_detected
        \n# Take in base64 string and return PIL image\ndef stringToImage(base64_string):\n    imgdata = base64.b64decode(base64_string)\n    return Image.open(io.BytesIO(imgdata))
        \n# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv\ndef toRGB(image):\n    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        \n@socketio.on("connect")\ndef handle_connect():\n    origin = request.headers.get("Origin")\n    print("New connection from origin:", origin)
        \n@socketio.on("image")\ndef handle_image(data):\n    try:\n        print("getting data.....")\n        base64Image = data["img"]
        \n        # Decode base64 image\n        image_decoded = stringToImage(base64Image)\n        img_colored = toRGB(image_decoded)
        \n        # Perform object detection\n        boxes, confidences, items = detect_objects(img_colored)
        \n        # Emit detected objects\n        emit(\n            "detected_objects",\n            {"boxes": boxes, "confidences": confidences, "items": items},\n        )
        \n    except Exception as e:\n        print("Error:", e)
        \nif __name__ == "__main__":\n    socketio.run(app, debug=True, host="localhost")`,
        lang: "python",
      },
    ],
    mobile: false,
    description: "",
    features: ["", "", ""],
    built_with: [
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
    ],
    published: true,
    date: "2024-04-02",
  },
  {
    slug: "tenant-portal",
    title: "Tenant App (Flutter)",
    overview:
      "Flutter mobile app for tenants, packed with essential features like maintenance requests, announcements, lease info, unit details, and tenant feedback. Empowering tenants with convenience and engagement, this app redefines the tenant experience with seamless functionality and user-friendly design.",
    links: {
      website: "",
      github: "tenant_portal",
    },
    images: [],
    code: [],
    mobile: true,
    description:
      "No screenshots available (Please check github repository for further details)",
    features: ["", "", ""],
    built_with: [
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
    ],
    published: true,
    date: "2023-02-10",
  },
  {
    slug: "websocket",
    title: "Websocket (Node.js)",
    overview:
      "This Node.js WebSocket server enables real-time communication between clients. Clients are assigned unique IDs upon connection, and messages, including image uploads, are broadcasted to all connected clients. Additionally, the server logs messages to a text file for reference, enhancing communication capabilities for applications requiring real-time updates.",
    links: {
      website: "",
      github: "websocket_nodejs",
    },
    images: [],
    code: [
      {
        title: "Real-Time Chat Application with WebSocket Logging in Node.js",
        description: `This Node.js project implements a real-time chat application using WebSocket technology. The application allows multiple users to connect simultaneously and exchange messages in real-time. Each user is assigned a unique identifier (UUID) upon connection, facilitating personalized interactions within the chat.

        Additionally, the application incorporates logging functionality to track user activities. Messages exchanged between users, including image uploads, are logged along with timestamps and user IDs. These logs are appended to a designated file (websocket_logs.txt) for future reference and analysis.
        
        The server utilizes the WebSocket module to establish bidirectional communication channels between the server and connected clients. When a client sends a message, it is broadcasted to all other connected clients, enabling seamless communication among users.
        
        This project showcases the power of WebSocket technology for building efficient real-time communication systems and demonstrates how to integrate logging functionality for monitoring and analyzing user interactions.`,
        features: [],
        code: `        const WebSocket = require("ws");
        const http = require("http");
        const fs = require("fs");
        const { v4: uuidv4 } = require("uuid");
        
        const server = http.createServer((req, res) => {
          res.writeHead(200, { "Content-Type": "text/plain" });
          res.end("WebSocket server is running");
        });
        
        const wss = new WebSocket.Server({ server });
        
        const loggingFilePath = "websocket_logs.txt";
        
        // Keep track of connected clients
        const clients = new Set();
        
        wss.on("connection", (ws, req) => {
          const userId = uuidv4();
          console.log('User#-userId connected');
        
          const clientIp = req.socket.remoteAddress;
          // Add the new client to the set
          clients.add(ws);
        
          // Event listener for receiving messages
          ws.on("message", (message) => {
            const messageObj = JSON.parse(message);
            if (messageObj.type == "image") {
              logMessage(userId, "sended image");
            } else {
              logMessage(userId, messageObj.message);
            }
        
            broadcast(
              JSON.stringify({
                user: userId,
                message: messageObj,
                client_ip: clientIp,
              })
            );
          });
        
          // Event listener for closing the connection
          ws.on("close", () => {
            console.log("Client disconnected");
        
            // Remove the disconnected client from the set
            clients.delete(ws);
          });
        });
        
        function logMessage(userId, message) {
          const logEntry = 'Date: UserID-Message';
        
          // Append the log entry to the file
          fs.appendFile(loggingFilePath, logEntry, (err) => {
            if (err) {
              console.error("Error writing to log file:", err);
            }
          });
        }
        
        // Broadcast a message to all connected clients
        function broadcast(message) {
          clients.forEach((client) => {
            // Check if the client is still connected before sending the message
            if (client.readyState === WebSocket.OPEN) {
              client.send(message);
            }
          });
        }
        
        const PORT = 3000;
        server.listen(PORT, () => {
          console.log('Server listening on http://localhost:PORT');
        });`,
        lang: "javascript",
      },
    ],
    mobile: false,
    description: "",
    features: ["", "", ""],
    built_with: [
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
    ],
    published: true,
    date: "2024-02-15",
  },
  {
    slug: "landing-pages",
    title: "Landing Pages (Front-End)",
    overview:
      "Introducing a series of captivating landing pages meticulously crafted with HTML, CSS and JavaScript (includes React.js/Next.js/Svelte). From sleek designs to vibrant layouts, each page is tailored to engage visitors and drive conversions. With seamless user experiences across devices, our landing pages are designed to leave a lasting impression and elevate your brand, event, or product launch.",
    links: {
      website: "",
      github: "",
    },
    images: [
      "landing_pages/img_1.png",
      "landing_pages/img_2.png",
      "landing_pages/img_3.png",
      "landing_pages/img_4.png",
      "landing_pages/img_5.png",
      "landing_pages/img_6.png",
      "landing_pages/img_7.png",
      "landing_pages/img_8.png",
      "landing_pages/img_9.png",
      "landing_pages/img_10.png",
    ],
    code: [],
    mobile: false,
    description: "",
    features: ["", "", ""],
    built_with: [
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
      {
        name: "",
        link: "",
      },
    ],
    published: true,
    date: "2022-10-15",
  },
];

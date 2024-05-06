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
      },
      {
        title:
          "Predictive Text Generation with Custom Language Models using TensorFlow",
        description:
          "This Python code showcases how to utilize a custom language model trained with TensorFlow for text generation. The script loads a pre-trained custom language model, tokenizes input text, pads sequences to a fixed length, and feeds them into the model for prediction. Specifically, it demonstrates the process of predicting the next token in a sequence using an LSTM layer. The code offers insights into reshaping input data for LSTM compatibility and obtaining predictions from the model's output layer. This can serve as a foundational guide for integrating and testing custom language models within TensorFlow applications.",
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
      },
      {
        title:
          "Text Generation Model Training from PDFs using TensorFlow and PyPDF2",
        description:
          "This Python script demonstrates the process of training a text generation model using PDF documents as a corpus. It begins by extracting text from a PDF file using PyPDF2, tokenizing the text, and converting it into sequences for model training. The script then pads the sequences to ensure uniform length, defines hyperparameters for the LSTM-based model, and compiles it with appropriate loss and optimizer settings. After splitting the data into training and validation sets, the model is trained and evaluated. Finally, the trained model is saved for future use. This code serves as a practical guide for building text generation models from PDF sources using TensorFlow and PyPDF2 libraries.",
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
      },
    ],
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
      "Introducing Flutter-based event management app, developed with expertise in cross-platform optimization. Streamlining event planning and attendance tracking, our app integrates QR scanning and backend functionalities in PHP, Node.js, and SQL. With user-friendly interfaces for iOS and Android, our solution ensures seamless navigation and widespread adoption, revolutionizing event management processes.",
    links: {
      website: "",
      github: "",
    },
    images: ["", "", "", ""],
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
      github: "",
    },
    images: ["", "", "", ""],
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
    date: "2022-12-27",
  },
  {
    slug: "image-optimization",
    title: "Sharp Image Optimization (Node.js)",
    overview:
      "Node.js backend application for image optimization, a powerful tool built with Sharp, Express, Multer, and Compression. Designed to enhance performance and reduce file sizes, this app optimizes images based on input parameters sent via API. From image dimensions and quality to export type and compression size, it ensures efficient optimization tailored to specific requirements. With seamless integration and robust functionality, it streamlines the image optimization process, empowering developers to enhance performance and user experience effortlessly.",
    links: {
      website: "",
      github: "",
    },
    images: ["", "", "", ""],
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
      github: "",
    },
    images: ["", "", "", ""],
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
      github: "",
    },
    images: ["", "", "", ""],
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
    date: "2023-02-10",
  },
  {
    slug: "websocket",
    title: "Websocket (Node.js)",
    overview:
      "This Node.js WebSocket server enables real-time communication between clients. Clients are assigned unique IDs upon connection, and messages, including image uploads, are broadcasted to all connected clients. Additionally, the server logs messages to a text file for reference, enhancing communication capabilities for applications requiring real-time updates.",
    links: {
      website: "",
      github: "",
    },
    images: ["", "", "", ""],
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
      "Introducing a series of captivating landing pages meticulously crafted with HTML and CSS. From sleek designs to vibrant layouts, each page is tailored to engage visitors and drive conversions. With seamless user experiences across devices, our landing pages are designed to leave a lasting impression and elevate your brand, event, or product launch.",
    links: {
      website: "",
      github: "",
    },
    images: ["", "", "", ""],
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

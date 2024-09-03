Here's the updated README with the site link added in the overview section:

# Paperbot: Chat with Websites

## Overview

Paperbot is an innovative interactive application that allows users to chat with websites using a conversational interface. By leveraging the power of Streamlit and OpenAI's language model, Paperbot provides intelligent, context-aware responses based on content extracted from specified websites. The primary objective is to create a seamless user experience where questions related to a website's content are answered accurately and meaningfully, mimicking a natural conversation.

The application is live and accessible at [https://paperbot-mjk4jthjyeiglyszdhnmsq.streamlit.app/](https://paperbot-mjk4jthjyeiglyszdhnmsq.streamlit.app/).

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
  - [Document Loading](#document-loading)
  - [Vector Store Creation](#vector-store-creation)
  - [Conversation Flow](#conversation-flow)
- [Deployment](#deployment)
- [Error Handling](#error-handling)
- [Contributing](#contributing)
- [License](#license)

## Key Features

- **Conversational Interface:** Users can chat with websites using a natural language interface, receiving context-aware answers.
- **Streamlit Integration:** The application is built using Streamlit, making it easy to deploy and accessible from any web browser.
- **LangChain Components:** Utilizes LangChain components to load, split, and process website content efficiently.
- **Vector Store:** Embeddings generated by OpenAI are stored in a vector store, enhancing the accuracy of responses by enabling context-aware retrieval.
- **History-Aware Retrieval:** The bot provides responses informed by the conversation's history, ensuring coherence and relevance.
- **User-Friendly Interface:** Simple and intuitive chat interface within Streamlit for smooth communication.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Clone the Repository

```bash
git clone https://github.com/yourusername/paperbot.git
cd paperbot
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### OpenAI API Key

Ensure you have an OpenAI API key. You can obtain one by signing up on the [OpenAI website](https://www.openai.com). Once you have the API key, create a `.env` file in the root directory of the project and add your API key as follows:

```bash
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Running the Application Locally

To run the application locally, use the following command:

```bash
streamlit run app.py
```

Once the application is running, open your web browser and navigate to `http://localhost:8501` to interact with Paperbot.

### Accessing the Deployed Application

Paperbot is also deployed on Streamlit and can be accessed online using the following link:

[https://paperbot-mjk4jthjyeiglyszdhnmsq.streamlit.app/](https://paperbot-mjk4jthjyeiglyszdhnmsq.streamlit.app/)

### Using the Chat Interface

1. **Enter a Website URL:** In the sidebar, input the URL of the website you wish to chat with.
2. **Ask Questions:** Type your questions in the chat interface, and the bot will respond in real-time.
3. **View Responses:** The bot provides context-aware answers based on the website's content.

## How It Works

### Document Loading

Paperbot uses LangChain components to load the content from the specified website. The content is then split into manageable chunks for efficient processing and storage.

### Vector Store Creation

Once the content is split, embeddings are generated using OpenAI's language model. These embeddings are stored in a vector store, which serves as the foundation for retrieving relevant information based on user queries.

### Conversation Flow

Paperbot employs a history-aware retrieval-augmented generation (RAG) chain. This feature ensures that the bot's responses are not only relevant to the current query but also informed by the history of the conversation, resulting in a more natural and coherent dialogue.

## Deployment

Paperbot is deployed on Streamlit and can be accessed through the following link:

[https://paperbot-mjk4jthjyeiglyszdhnmsq.streamlit.app/](https://paperbot-mjk4jthjyeiglyszdhnmsq.streamlit.app/)

This deployment ensures that the application is easily accessible online for testing and use.

## Error Handling

Paperbot includes error handling for scenarios such as missing or incorrectly formatted URLs. If an error occurs, the application provides a user-friendly message guiding the user to correct the issue.

### Common Errors

- **Invalid URL:** If the provided URL is invalid, the application will notify the user and prompt them to enter a valid URL.
- **Empty Response:** If the website's content cannot be retrieved, the application will inform the user and suggest checking the website's accessibility.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

This README provides an overview of the Paperbot project, including installation instructions, usage guidelines, and details on how the application works. For any further questions or support, feel free to open an issue or contact the repository maintainers.

Flowchart :
![flowchart](https://github.com/user-attachments/assets/fbea9218-c553-4ed7-8727-77f193849dcb)

![Home](https://github.com/user-attachments/assets/e6d1ba77-8df1-4f60-ac53-447621f7e98c)

<img width="1440" alt="pic1" src="https://github.com/user-attachments/assets/f98d12ae-3fc8-4756-a4e2-208229dcd389">


<img width="1440" alt="pic2" src="https://github.com/user-attachments/assets/fbbf3578-6ed5-4902-9c28-488a0fef0a94">





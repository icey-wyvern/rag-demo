# UK GVT RAG Demo

A simple demo for asking questions about a set of seeded GOV.UK documents using RAG and the OpenAI API.

---

## Setup

1. Clone this repository and open it in a terminal.

2. In the project root, edit `.env` and add your OpenAI API key:

   ```env
   OPENAI_API_KEY=sk-your-key-here
   ```

3. From the project root, run:

   ```bash
   cd /path/to/rag-demo
   docker-compose up --build
   ```

---

## Running the app

Once Docker has finished starting up, open:

```text
http://localhost:8080
```

---

## Using the app

* Type a question into the input box.
* Choose **Generative** or **Chunks only**.
* Click **Ask**.
* Use the source links to open the underlying PDFs (they jump to the relevant page).

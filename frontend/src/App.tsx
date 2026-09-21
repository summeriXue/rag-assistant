import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

type RagResult = {
  answer: string
  sources: string[]
  trace: {
    original_question: string
    final_search_query: string
    retrieved_count: number
    reranked_count: number
    documents_relevant: boolean
    answer_supported: boolean
    retrieval_retry_count: number
    generation_retry_count: number
  }
}

function App() {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState<RagResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleAsk() {
    if (!question.trim()) {
      return
    }

    setLoading(true)
    setResult(null)
    setError('')

    try {
      const response = await fetch('http://127.0.0.1:8000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question,
        }),
      })

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`)
      }

      const data = await response.json()

      setResult(data)
    } catch (error) {
      console.error(error)
      setError('Failed to get an answer. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main>
      <h1>Agentic RAG Assistant</h1>
      <p>Ask questions about the knowledge base.</p>

      <textarea
        value={question}
        onChange={(event) => setQuestion(event.target.value)}
        placeholder="Ask a question..."
        rows={4}
      />

      <button
        type="button"
        onClick={handleAsk}
        disabled={loading}
      >
        {loading ? 'Thinking...' : 'Ask'}
      </button>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {result && (
        <section>
          <h2>Answer</h2>
          <div className="answer-card">
            <ReactMarkdown>{result.answer}</ReactMarkdown>
          </div>

          <h2>Sources</h2>
          <ul>
            {result.sources.map((source) => (
              <li key={source}>{source}</li>
            ))}
          </ul>

          <details>
            <summary>Retrieval Trace</summary>

            <div className="trace-content">
              <div className="trace-query">
                <span className="trace-label">Original Question</span>
                <span>{result.trace.original_question}</span>
              </div>

              <div className="trace-query">
                <span className="trace-label">Transformed Query</span>
                <span>{result.trace.final_search_query}</span>
              </div>

              <div className="pipeline">
                <div className="pipeline-step">
                  <span className="step-name">Transform</span>
                  <span className="step-status">DONE</span>
                </div>

                <span className="pipeline-arrow">→</span>

                <div className="pipeline-step">
                  <span className="step-name">Retrieve</span>
                  <span className="step-value">
                    {result.trace.retrieved_count} candidates
                  </span>
                </div>

                <span className="pipeline-arrow">→</span>

                <div className="pipeline-step">
                  <span className="step-name">Rerank</span>
                  <span className="step-value">
                    Top {result.trace.reranked_count}
                  </span>
                </div>

                <span className="pipeline-arrow">→</span>

                <div className="pipeline-step">
                  <span className="step-name">Grade</span>
                  <span
                    className={
                      result.trace.documents_relevant
                        ? 'step-status pass'
                        : 'step-status fail'
                    }
                  >
                    {result.trace.documents_relevant ? 'PASS' : 'FAIL'}
                  </span>
                </div>

                <span className="pipeline-arrow">→</span>

                <div className="pipeline-step">
                  <span className="step-name">Generate</span>
                  <span className="step-status">DONE</span>
                </div>

                <span className="pipeline-arrow">→</span>

                <div className="pipeline-step">
                  <span className="step-name">Verify</span>
                  <span
                    className={
                      result.trace.answer_supported
                        ? 'step-status pass'
                        : 'step-status fail'
                    }
                  >
                    {result.trace.answer_supported ? 'PASS' : 'FAIL'}
                  </span>
                </div>
              </div>

              <div className="retry-row">
                <span>
                  Retrieval retries:
                  <strong> {result.trace.retrieval_retry_count}</strong>
                </span>

                <span>
                  Generation retries:
                  <strong> {result.trace.generation_retry_count}</strong>
                </span>
              </div>
            </div>
          </details>
        </section>
      )}
    </main>
  )
}

export default App

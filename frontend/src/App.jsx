import { useState, useCallback, useEffect } from 'react'
import './App.css'

// Backend API URL resolution order:
// 1) Runtime config: ./config.json (editable after build)
// 2) Build-time env: VITE_API_URL
// 3) Same-host fallback: <current-origin>/api
const API_URL_FALLBACK = import.meta.env?.VITE_API_URL || `${window.location.origin}/api`

// Language options
const LANGUAGES = {
  ar: { name: 'Tiếng Ả Rập', flag: '🇸🇦' },
  vi: { name: 'Tiếng Việt', flag: '🇻🇳' },
  en: { name: 'Tiếng Anh', flag: '🇺🇸' },
  'de-DE': { name: 'Tiếng Đức', flag: '🇩🇪' },
  cs: { name: 'Tiếng Séc', flag: '🇨🇿' },
  fr: { name: 'Tiếng Pháp', flag: '🇫🇷' },
  es: { name: 'Tiếng Tây Ban Nha', flag: '🇪🇸' },
  zh: { name: 'Tiếng Trung', flag: '🇨🇳' },
  ja: { name: 'Tiếng Nhật', flag: '🇯🇵' },
  ko: { name: 'Tiếng Hàn', flag: '🇰🇷' },
  ru: { name: 'Tiếng Nga', flag: '🇷🇺' },
  th: { name: 'Tiếng Thái', flag: '🇹🇭' },
}

function App() {
  // Tab state
  const [activeTab, setActiveTab] = useState('csv')

  // Runtime-configurable API URL
  const [apiUrl, setApiUrl] = useState(API_URL_FALLBACK)

  // Translation settings - Default to Gemma
  const [method, setMethod] = useState('gemma')
  const [sourceLang, setSourceLang] = useState('ar')
  const [targetLang, setTargetLang] = useState('vi')

  // CSV translation state
  const [file, setFile] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(null)

  // Text translation state
  const [inputText, setInputText] = useState('')
  const [translatedText, setTranslatedText] = useState('')
  const [isTranslating, setIsTranslating] = useState(false)

  // Image translation state
  const [imageFile, setImageFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [imageResult, setImageResult] = useState('')
  const [isImageTranslating, setIsImageTranslating] = useState(false)

  // Load runtime config once (optional)
  useEffect(() => {
    let cancelled = false

    async function loadRuntimeConfig() {
      try {
        // In Vite build, assets are served from base URL. config.json is in /config.json
        const res = await fetch('./config.json', { cache: 'no-store' })
        if (!res.ok) return
        const cfg = await res.json()
        if (cancelled) return

        if (cfg && typeof cfg.apiUrl === 'string' && cfg.apiUrl.trim()) {
          setApiUrl(cfg.apiUrl.trim())
        }
      } catch {
        // ignore, keep fallback
      }
    }

    loadRuntimeConfig()
    return () => {
      cancelled = true
    }
  }, [])

  const handleDrag = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0]
      if (droppedFile.name.endsWith('.csv')) {
        setFile(droppedFile)
        setError(null)
      } else {
        setError('Vui lòng chọn file CSV')
      }
    }
  }, [])

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setError(null)
    }
  }

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      setImageFile(file)
      setImageResult('')

      // Create preview
      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const uploadFile = async () => {
    if (!file) return

    setError(null)
    setStatus({ status: 'uploading', message: 'Đang tải file lên...' })

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(
        `${apiUrl}/upload?method=${method}&source_lang=${sourceLang}&target_lang=${targetLang}`,
        {
          method: 'POST',
          body: formData,
        }
      )

      if (!response.ok) {
        throw new Error('Upload failed')
      }

      const data = await response.json()
      setJobId(data.job_id)
      setStatus({
        status: 'queued',
        progress: 0,
        queue_position: data.queue_position ?? 0,
        message: 'Đã nhận file, đang đưa vào hàng đợi...'
      })
      pollStatus(data.job_id)
    } catch (err) {
      setError('Lỗi khi tải file: ' + err.message)
      setStatus(null)
    }
  }

  const pollStatus = async (id) => {
    const poll = async () => {
      try {
        const response = await fetch(`${apiUrl}/status/${id}`)
        const data = await response.json()
        setStatus(data)

        if (data.status === 'queued' || data.status === 'processing') {
          setTimeout(poll, 1000)
        } else if (data.status === 'error') {
          setError(data.error)
        }
      } catch (err) {
        setError('Lỗi kết nối: ' + err.message)
      }
    }
    poll()
  }

  const downloadFile = () => {
    if (jobId) {
      window.open(`${apiUrl}/download/${jobId}`, '_blank')
    }
  }

  const reset = () => {
    setFile(null)
    setJobId(null)
    setStatus(null)
    setError(null)
  }

  const translateText = async () => {
    if (!inputText.trim()) return

    setIsTranslating(true)
    setTranslatedText('')

    try {
      const response = await fetch(`${apiUrl}/translate-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          source_lang: sourceLang,
          target_lang: targetLang,
          method: method
        })
      })

      const data = await response.json()
      setTranslatedText(data.translated_text)
    } catch (err) {
      setError('Lỗi dịch: ' + err.message)
    } finally {
      setIsTranslating(false)
    }
  }

  const translateImage = async () => {
    if (!imageFile) return

    setIsImageTranslating(true)
    setImageResult('')

    try {
      // Convert to base64
      const reader = new FileReader()
      reader.onloadend = async () => {
        const base64 = reader.result.split(',')[1]

        const response = await fetch(`${apiUrl}/translate-image`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: base64,
            source_lang: sourceLang,
            target_lang: targetLang
          })
        })

        const data = await response.json()
        setImageResult(data.translated_text)
        setIsImageTranslating(false)
      }
      reader.readAsDataURL(imageFile)
    } catch (err) {
      setError('Lỗi dịch ảnh: ' + err.message)
      setIsImageTranslating(false)
    }
  }

  const swapLanguages = () => {
    setSourceLang(targetLang)
    setTargetLang(sourceLang)
  }

  return (
    <div className="app">
      <div className="container">
        <div className="header">
          <h1>CSV Translator Pro</h1>
          <p>Dịch văn bản đa ngôn ngữ với AI</p>
        </div>

        {/* Tabs */}
        <div className="tabs">
          <button
            className={`tab ${activeTab === 'csv' ? 'active' : ''}`}
            onClick={() => setActiveTab('csv')}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14,2 14,8 20,8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
            </svg>
            CSV
          </button>
          <button
            className={`tab ${activeTab === 'text' ? 'active' : ''}`}
            onClick={() => setActiveTab('text')}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            Text
          </button>
          <button
            className={`tab ${activeTab === 'image' ? 'active' : ''}`}
            onClick={() => setActiveTab('image')}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <polyline points="21,15 16,10 5,21" />
            </svg>
            Ảnh/OCR
          </button>
        </div>

        {/* Translation Method Selector - Hidden, default to Gemma */}
        {/* <div className="method-selector">
          <span className="label">Engine:</span>
          <div className="method-options">
            <button
              className={`method-btn ${method === 'helsinki' ? 'active' : ''}`}
              onClick={() => setMethod('helsinki')}
            >
              <span className="method-icon">⚡</span>
              <span className="method-name">Helsinki-NLP</span>
              <span className="method-desc">Nhanh</span>
            </button>
            <button
              className={`method-btn ${method === 'gemma' ? 'active' : ''}`}
              onClick={() => setMethod('gemma')}
            >
              <span className="method-icon">🧠</span>
              <span className="method-name">TranslateGemma</span>
              <span className="method-desc">Chính xác</span>
            </button>
          </div>
        </div> */}

        {/* Language Selector */}
        <div className="language-selector">
          <div className="lang-dropdown">
            <label>Nguồn</label>
            <select value={sourceLang} onChange={(e) => setSourceLang(e.target.value)}>
              {Object.entries(LANGUAGES).map(([code, lang]) => (
                <option key={code} value={code}>{lang.flag} {lang.name}</option>
              ))}
            </select>
          </div>
          <button className="swap-btn" onClick={swapLanguages}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="17,1 21,5 17,9" />
              <path d="M3 11V9a4 4 0 0 1 4-4h14" />
              <polyline points="7,23 3,19 7,15" />
              <path d="M21 13v2a4 4 0 0 1-4 4H3" />
            </svg>
          </button>
          <div className="lang-dropdown">
            <label>Đích</label>
            <select value={targetLang} onChange={(e) => setTargetLang(e.target.value)}>
              {Object.entries(LANGUAGES).map(([code, lang]) => (
                <option key={code} value={code}>{lang.flag} {lang.name}</option>
              ))}
            </select>
          </div>
        </div>

        {/* CSV Tab Content */}
        {activeTab === 'csv' && (
          <>
            {!status && (
              <div
                className={`dropzone ${dragActive ? 'active' : ''} ${file ? 'has-file' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <input
                  type="file"
                  id="file-input"
                  accept=".csv"
                  onChange={handleFileChange}
                  hidden
                />
                <label htmlFor="file-input" className="dropzone-content">
                  {file ? (
                    <>
                      <span className="file-icon">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                          <polyline points="14,2 14,8 20,8" />
                        </svg>
                      </span>
                      <span className="file-name">{file.name}</span>
                      <span className="file-size">{(file.size / 1024).toFixed(1)} KB</span>
                    </>
                  ) : (
                    <>
                      <span className="upload-icon">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                          <polyline points="17,8 12,3 7,8" />
                          <line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                      </span>
                      <span>Kéo thả file CSV vào đây</span>
                      <span className="or">hoặc click để chọn file</span>
                    </>
                  )}
                </label>
              </div>
            )}

            {file && !status && (
              <button className="btn btn-primary" onClick={uploadFile}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="5,3 19,12 5,21 5,3" />
                </svg>
                Bắt đầu dịch
              </button>
            )}

            {status && status.status === 'queued' && (
              <div className="progress-container">
                <div className="spinner"></div>
                <p>
                  {status.message}
                  {Number.isFinite(status.queue_position) && status.queue_position > 0
                    ? ` (vị trí hàng đợi: ${status.queue_position})`
                    : ''}
                </p>
              </div>
            )}

            {status && status.status === 'processing' && (
              <div className="progress-container">
                <div className="progress-info">
                  <span>{status.message}</span>
                  <span>{status.progress}%</span>
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${status.progress}%` }}
                  />
                </div>
              </div>
            )}

            {status && status.status === 'uploading' && (
              <div className="progress-container">
                <div className="spinner"></div>
                <p>{status.message}</p>
              </div>
            )}

            {status && status.status === 'completed' && (
              <div className="success-container">
                <div className="success-icon">
                  <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                    <polyline points="22,4 12,14.01 9,11.01" />
                  </svg>
                </div>
                <h2>Dịch hoàn thành!</h2>
                <p>File CSV đã được dịch thành công</p>
                <div className="btn-group">
                  <button className="btn btn-success" onClick={downloadFile}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="7,10 12,15 17,10" />
                      <line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                    Tải file đã dịch
                  </button>
                  <button className="btn btn-secondary" onClick={reset}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="23,4 23,10 17,10" />
                      <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
                    </svg>
                    Dịch file khác
                  </button>
                </div>
              </div>
            )}
          </>
        )}

        {/* Text Tab Content */}
        {activeTab === 'text' && (
          <div className="text-translate-container">
            <div className="text-input-group">
              <textarea
                className="text-input"
                placeholder="Nhập văn bản cần dịch..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                rows={5}
              />
              <button
                className="btn btn-primary"
                onClick={translateText}
                disabled={isTranslating || !inputText.trim()}
              >
                {isTranslating ? (
                  <>
                    <div className="spinner-small"></div>
                    Đang dịch...
                  </>
                ) : (
                  <>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M5 12h14M12 5l7 7-7 7" />
                    </svg>
                    Dịch
                  </>
                )}
              </button>
            </div>
            {translatedText && (
              <div className="translated-result">
                <label>Kết quả:</label>
                <div className="result-text">{translatedText}</div>
              </div>
            )}
          </div>
        )}

        {/* Image Tab Content */}
        {activeTab === 'image' && (
          <div className="image-translate-container">
            <div className="image-upload-area">
              <input
                type="file"
                id="image-input"
                accept="image/*"
                onChange={handleImageChange}
                hidden
              />
              <label htmlFor="image-input" className="image-dropzone">
                {imagePreview ? (
                  <img src={imagePreview} alt="Preview" className="image-preview" />
                ) : (
                  <>
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                      <circle cx="8.5" cy="8.5" r="1.5" />
                      <polyline points="21,15 16,10 5,21" />
                    </svg>
                    <span>Click để chọn ảnh</span>
                    <span className="or">Hỗ trợ JPG, PNG, WebP</span>
                  </>
                )}
              </label>
            </div>

            {imageFile && (
              <button
                className="btn btn-primary"
                onClick={translateImage}
                disabled={isImageTranslating}
              >
                {isImageTranslating ? (
                  <>
                    <div className="spinner-small"></div>
                    Đang xử lý ảnh...
                  </>
                ) : (
                  <>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M5 12h14M12 5l7 7-7 7" />
                    </svg>
                    Trích xuất & Dịch
                  </>
                )}
              </button>
            )}

            {imageResult && (
              <div className="translated-result">
                <label>Văn bản trích xuất & dịch:</label>
                <div className="result-text">{imageResult}</div>
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="error-container">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="15" y1="9" x2="9" y2="15" />
              <line x1="9" y1="9" x2="15" y2="15" />
            </svg>
            <span>{error}</span>
            <button className="btn btn-secondary" onClick={() => setError(null)}>
              Đóng
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default App

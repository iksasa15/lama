import { useCallback, useEffect, useId, useRef, useState } from 'react'
import './App.css'

const MODELS = [
  { id: 'people', label: 'عدّ الأشخاص', color: '#2dd4bf' },
  { id: 'smoke', label: 'كشف الدخان', color: '#94a3b8' },
  { id: 'fire', label: 'كشف الحريق', color: '#f97316' },
  { id: 'fall', label: 'سقوط / طوارئ', color: '#ef4444' },
  { id: 'wrong_way', label: 'عكس السير', color: '#eab308' },
]

const DIRECTIONS = [
  { id: 'up', label: '↑ أعلى', hint: 'من الأسفل إلى الأعلى' },
  { id: 'down', label: '↓ أسفل', hint: 'من الأعلى إلى الأسفل' },
  { id: 'left', label: '← يسار', hint: 'من اليمين إلى اليسار' },
  { id: 'right', label: '→ يمين', hint: 'من اليسار إلى اليمين' },
]

const COLOR_BY_MODEL = Object.fromEntries(MODELS.map((m) => [m.id, m.color]))

function createSessionId() {
  return `sess-${Math.random().toString(36).slice(2, 10)}`
}

export default function App() {
  const sessionId = useRef(createSessionId()).current
  const videoRef = useRef(null)
  const imageRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const busyRef = useRef(false)
  const fileInputId = useId()

  const [enabled, setEnabled] = useState(() =>
    Object.fromEntries(MODELS.map((m) => [m.id, false])),
  )
  const [sourceMode, setSourceMode] = useState('idle') // idle | webcam | image | video
  const [status, setStatus] = useState('اختر موديلات وفعّل الكاميرا أو ارفع ملفاً')
  const [error, setError] = useState('')
  const [peopleCount, setPeopleCount] = useState(0)
  const [alerts, setAlerts] = useState([])
  const [detections, setDetections] = useState([])
  const [running, setRunning] = useState(false)
  const [correctDirection, setCorrectDirection] = useState('down')

  const enabledList = MODELS.filter((m) => enabled[m.id]).map((m) => m.id)

  const toggleModel = (id) => {
    setEnabled((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
  }, [])

  const clearCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
  }

  const drawDetections = useCallback((dets, width, height) => {
    const canvas = canvasRef.current
    if (!canvas) return
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, width, height)
    for (const d of dets) {
      const [x1, y1, x2, y2] = d.box
      const color = COLOR_BY_MODEL[d.model] || '#38bdf8'
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
      const text = `${d.label} ${(d.score * 100).toFixed(0)}%`
      ctx.font = '14px "IBM Plex Sans Arabic", "Segoe UI", sans-serif'
      const tw = ctx.measureText(text).width
      ctx.fillStyle = color
      ctx.fillRect(x1, Math.max(0, y1 - 20), tw + 8, 20)
      ctx.fillStyle = '#0b1220'
      ctx.fillText(text, x1 + 4, Math.max(14, y1 - 5))
    }
  }, [])

  const sendFrame = useCallback(
    async (blob) => {
      if (!blob || enabledList.length === 0 || busyRef.current) return
      busyRef.current = true
      setStatus('جاري التحليل...')
      setError('')
      try {
        const form = new FormData()
        form.append('image', blob, 'frame.jpg')
        form.append('session_id', sessionId)
        form.append('enabled', JSON.stringify(enabledList))
        form.append('correct_direction', correctDirection)
        const res = await fetch('/api/detect', { method: 'POST', body: form })
        if (!res.ok) {
          const text = await res.text()
          throw new Error(text || `HTTP ${res.status}`)
        }
        const data = await res.json()
        setPeopleCount(data.people_count ?? 0)
        setAlerts(data.alerts ?? [])
        setDetections(data.detections ?? [])

        let w = 640
        let h = 480
        if (sourceMode === 'webcam' || sourceMode === 'video') {
          const v = videoRef.current
          if (v) {
            w = v.videoWidth || w
            h = v.videoHeight || h
          }
        } else if (sourceMode === 'image' && imageRef.current) {
          w = imageRef.current.naturalWidth || w
          h = imageRef.current.naturalHeight || h
        }
        drawDetections(data.detections ?? [], w, h)
        setStatus('تم')
      } catch (err) {
        setError(
          err.message === 'Failed to fetch' || /ECONNREFUSED|proxy/i.test(err.message)
            ? 'السيرفر غير متصل — شغّل: npm run dev:all'
            : err.message || 'فشل الطلب — تأكد أن سيرفر بايثون يعمل',
        )
        setStatus('خطأ')
      } finally {
        busyRef.current = false
      }
    },
    [enabledList, sessionId, sourceMode, drawDetections, correctDirection],
  )

  const captureFromVideo = useCallback(() => {
    const video = videoRef.current
    if (!video || video.readyState < 2) return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0)
    canvas.toBlob((blob) => sendFrame(blob), 'image/jpeg', 0.92)
  }, [sendFrame])

  const captureFromImage = useCallback(() => {
    const img = imageRef.current
    if (!img || !img.complete) return
    const canvas = document.createElement('canvas')
    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(img, 0, 0)
    canvas.toBlob((blob) => sendFrame(blob), 'image/jpeg', 0.92)
  }, [sendFrame])

  // Poll webcam / video frames
  useEffect(() => {
    if (!running || enabledList.length === 0) return
    if (sourceMode !== 'webcam' && sourceMode !== 'video') return
    const id = setInterval(captureFromVideo, 700)
    return () => clearInterval(id)
  }, [running, enabledList.length, sourceMode, captureFromVideo])

  // Analyze still image when ready / models change
  useEffect(() => {
    if (sourceMode !== 'image' || !running || enabledList.length === 0) return
    captureFromImage()
  }, [sourceMode, running, enabledList.join(','), captureFromImage])

  const startWebcam = async () => {
    setError('')
    stopCamera()
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setSourceMode('webcam')
      setRunning(true)
      setStatus('الكاميرا تعمل')
      clearCanvas()
    } catch (err) {
      setError('تعذر فتح الكاميرا: ' + (err.message || err))
    }
  }

  const onFileChange = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    stopCamera()
    clearCanvas()
    setAlerts([])
    setDetections([])
    setPeopleCount(0)

    if (file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file)
      if (imageRef.current) {
        imageRef.current.onload = () => {
          setSourceMode('image')
          setRunning(true)
          setStatus('صورة جاهزة')
        }
        imageRef.current.src = url
      }
    } else if (file.type.startsWith('video/')) {
      const url = URL.createObjectURL(file)
      if (videoRef.current) {
        videoRef.current.srcObject = null
        videoRef.current.src = url
        videoRef.current.onloadeddata = async () => {
          await videoRef.current.play()
          setSourceMode('video')
          setRunning(true)
          setStatus('فيديو يعمل')
        }
      }
    } else {
      setError('نوع الملف غير مدعوم')
    }
    e.target.value = ''
  }

  const stopAll = () => {
    setRunning(false)
    stopCamera()
    if (videoRef.current) {
      videoRef.current.pause()
      videoRef.current.removeAttribute('src')
      videoRef.current.load()
    }
    if (imageRef.current) {
      imageRef.current.removeAttribute('src')
    }
    setSourceMode('idle')
    clearCanvas()
    setStatus('متوقف')
  }

  useEffect(() => () => stopCamera(), [stopCamera])

  return (
    <div className="app" dir="rtl">
      <header className="header">
        <h1>لوحة الكشف المتعدد</h1>
        <p className="subtitle">
          موديلات YOLO من Hugging Face — شغّل أكثر من كاشف في نفس الوقت
        </p>
      </header>

      <section className="models">
        <h2>اختر الموديلات</h2>
        <div className="model-grid">
          {MODELS.map((m) => (
            <button
              key={m.id}
              type="button"
              className={`model-btn ${enabled[m.id] ? 'on' : ''}`}
              style={{ '--accent': m.color }}
              onClick={() => toggleModel(m.id)}
            >
              <span className="dot" />
              {m.label}
            </button>
          ))}
        </div>
        {enabled.wrong_way && (
          <div className="direction-picker">
            <h3>اتجاه السير الصحيح</h3>
            <p className="direction-hint">
              اختر اتجاه الحركة المسموح في الكاميرا — أي مركبة عكسه تُنبَّه
            </p>
            <div className="direction-grid">
              {DIRECTIONS.map((d) => (
                <button
                  key={d.id}
                  type="button"
                  className={`direction-btn ${correctDirection === d.id ? 'on' : ''}`}
                  title={d.hint}
                  onClick={() => setCorrectDirection(d.id)}
                >
                  {d.label}
                </button>
              ))}
            </div>
          </div>
        )}
      </section>

      <section className="controls">
        <button type="button" className="primary" onClick={startWebcam}>
          فتح الكاميرا
        </button>
        <label className="primary file-btn" htmlFor={fileInputId}>
          رفع صورة / فيديو
        </label>
        <input
          id={fileInputId}
          type="file"
          accept="image/*,video/*"
          hidden
          onChange={onFileChange}
        />
        <button type="button" className="ghost" onClick={stopAll}>
          إيقاف
        </button>
        {sourceMode === 'image' && (
          <button
            type="button"
            className="ghost"
            onClick={captureFromImage}
            disabled={enabledList.length === 0}
          >
            تحليل الصورة الآن
          </button>
        )}
      </section>

      <div className="workspace">
        <div className="stage">
          <video
            ref={videoRef}
            className={
              sourceMode === 'webcam' || sourceMode === 'video' ? 'media show' : 'media'
            }
            playsInline
            muted
            loop={sourceMode === 'video'}
          />
          <img
            ref={imageRef}
            alt=""
            className={sourceMode === 'image' ? 'media show' : 'media'}
          />
          <canvas ref={canvasRef} className="overlay" />
          {sourceMode === 'idle' && (
            <div className="placeholder">لا يوجد مصدر بعد</div>
          )}
        </div>

        <aside className="panel">
          <div className="stat">
            <span className="stat-label">عدد الأشخاص</span>
            <span className="stat-value">{peopleCount}</span>
          </div>
          <div className="stat">
            <span className="stat-label">الحالة</span>
            <span className="stat-value small">{status}</span>
          </div>
          <div className="alerts">
            <h3>التنبيهات</h3>
            {alerts.length === 0 ? (
              <p className="muted">لا تنبيهات</p>
            ) : (
              <ul>
                {alerts.map((a) => (
                  <li key={a} className={`alert alert-${a}`}>
                    {a === 'fire' && 'حريق'}
                    {a === 'smoke' && 'دخان'}
                    {a === 'fall' && 'سقوط'}
                    {a === 'wrong_way' && 'عكس السير'}
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="det-list">
            <h3>الكشوفات ({detections.length})</h3>
            <ul>
              {detections.slice(0, 12).map((d, i) => (
                <li key={`${d.label}-${i}`}>
                  <span
                    className="chip"
                    style={{ background: COLOR_BY_MODEL[d.model] || '#64748b' }}
                  />
                  {d.label} — {(d.score * 100).toFixed(0)}%
                </li>
              ))}
            </ul>
          </div>
          {error && <p className="error">{error}</p>}
          {enabledList.length === 0 && (
            <p className="hint">فعّل موديلاً واحداً على الأقل من الأزرار أعلاه</p>
          )}
        </aside>
      </div>
    </div>
  )
}

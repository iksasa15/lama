import { startTransition, useCallback, useEffect, useId, useRef, useState } from 'react'
import './App.css'

const MODELS = [
  { id: 'people', label: 'عدّ الأشخاص (YOLO)', color: '#2dd4bf' },
  { id: 'dmcount', label: 'عد الحشود (DM-Count)', color: '#a78bfa' },
  { id: 'smoke', label: 'كشف الدخان', color: '#94a3b8' },
  { id: 'fire', label: 'كشف الحريق', color: '#f97316' },
  { id: 'fall', label: 'سقوط / طوارئ', color: '#ef4444' },
  { id: 'wrong_way', label: 'اتجاه الأشخاص', color: '#eab308' },
]

const DIRECTIONS = [
  { id: 'up', arrow: '↑', label: 'أعلى' },
  { id: 'down', arrow: '↓', label: 'أسفل' },
  { id: 'left', arrow: '←', label: 'يسار' },
  { id: 'right', arrow: '→', label: 'يمين' },
]

const COLOR_BY_MODEL = {
  people: '#2dd4bf',
  dmcount: '#a78bfa',
  smoke: '#94a3b8',
  fire: '#f97316',
  fall: '#ef4444',
  wrong_way: '#ef4444',
  ok_way: '#22c55e',
  tracking: '#eab308',
}

function boxColor(d) {
  if (d.status === 'ok' || d.model === 'ok_way' || d.label === 'صح' || d.label === 'OK') return '#22c55e'
  if (d.status === 'wrong' || d.model === 'wrong_way' || d.label === 'غلط' || d.label === 'Wrong way') return '#ef4444'
  if (d.status === 'unknown' || d.model === 'tracking' || d.label === 'تتبع' || d.label === 'Tracking') return '#eab308'
  if (d.label === 'fallen' || d.label === 'Fallen') return '#ef4444'
  if (d.label === 'sitting' || d.label === 'Sitting') return '#f59e0b'
  if (d.label === 'standing' || d.label === 'Standing') return '#22c55e'
  return COLOR_BY_MODEL[d.model] || '#38bdf8'
}

function displayLabel(d) {
  const map = {
    fallen: 'Fallen',
    sitting: 'Sitting',
    standing: 'Standing',
    fire: 'Fire',
    smoke: 'Smoke',
    person: 'Person',
    صح: 'OK',
    غلط: 'Wrong way',
    تتبع: 'Tracking',
  }
  if (typeof d.label === 'string' && d.label.startsWith('Crowd')) return d.label
  return map[d.label] || d.label
}

function createSessionId() {
  return `live-${Math.random().toString(36).slice(2, 10)}`
}

/** Letterbox rect matching CSS object-fit: contain */
function containRect(nw, nh, cw, ch) {
  const scale = Math.min(cw / nw, ch / nh)
  const dw = nw * scale
  const dh = nh * scale
  return {
    scale,
    ox: (cw - dw) / 2,
    oy: (ch - dh) / 2,
    dw,
    dh,
  }
}

function boxIou(a, b) {
  const x1 = Math.max(a[0], b[0])
  const y1 = Math.max(a[1], b[1])
  const x2 = Math.min(a[2], b[2])
  const y2 = Math.min(a[3], b[3])
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  if (inter <= 0) return 0
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1])
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1])
  return inter / (areaA + areaB - inter + 1e-6)
}

function lerpBox(from, to, t) {
  return [
    from[0] + (to[0] - from[0]) * t,
    from[1] + (to[1] - from[1]) * t,
    from[2] + (to[2] - from[2]) * t,
    from[3] + (to[3] - from[3]) * t,
  ]
}

const TRACK_HOLD_MS = 1000
const LERP_SPEED = 0.35
const IOU_MATCH = 0.25

export default function App() {
  const sessionIdRef = useRef(createSessionId())
  const videoRef = useRef(null)
  const imageRef = useRef(null)
  const canvasRef = useRef(null)
  const stageRef = useRef(null)
  const streamRef = useRef(null)
  const busyRef = useRef(false)
  const captureCanvasRef = useRef(null)
  const lastDrawRef = useRef({ dets: [], w: 0, h: 0 })
  const canvasSizeRef = useRef({ w: 0, h: 0, dpr: 1 })
  const tracksRef = useRef(new Map()) // id -> { box, target, color, lastSeen, meta }
  const frameSizeRef = useRef({ w: 640, h: 480 })
  const tempIdRef = useRef(1)
  const enabledRef = useRef({})
  const sourceModeRef = useRef('idle')
  const directionRef = useRef('down')
  const fileInputId = useId()

  const [enabled, setEnabled] = useState(() =>
    Object.fromEntries(MODELS.map((m) => [m.id, m.id === 'people'])),
  )
  const [sourceMode, setSourceMode] = useState('idle') // idle | webcam | image | video
  const [status, setStatus] = useState('اختر موديلات وفعّل الكاميرا أو ارفع ملفاً')
  const [error, setError] = useState('')
  const [peopleCount, setPeopleCount] = useState(0)
  const [alerts, setAlerts] = useState([])
  const [detections, setDetections] = useState([])
  const [running, setRunning] = useState(false)
  const [correctDirection, setCorrectDirection] = useState('down')

  enabledRef.current = enabled
  sourceModeRef.current = sourceMode
  directionRef.current = correctDirection

  const enabledList = MODELS.filter((m) => enabled[m.id]).map((m) => m.id)
  const enabledKey = enabledList.join(',')

  const changeDirection = (id) => {
    if (id === correctDirection) return
    setCorrectDirection(id)
    sessionIdRef.current = createSessionId()
    tracksRef.current.clear()
    setAlerts([])
    setDetections([])
  }

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

  const clearTracks = useCallback(() => {
    tracksRef.current.clear()
    clearCanvas()
  }, [])

  const paintTracks = useCallback(() => {
    const canvas = canvasRef.current
    const stage = stageRef.current
    const { w: naturalW, h: naturalH } = frameSizeRef.current
    if (!canvas || !stage || !naturalW || !naturalH) return

    const cw = Math.max(1, stage.clientWidth)
    const ch = Math.max(1, stage.clientHeight)
    const dpr = window.devicePixelRatio || 1
    const needResize =
      canvasSizeRef.current.w !== cw ||
      canvasSizeRef.current.h !== ch ||
      canvasSizeRef.current.dpr !== dpr

    if (needResize) {
      canvas.width = Math.round(cw * dpr)
      canvas.height = Math.round(ch * dpr)
      canvas.style.width = `${cw}px`
      canvas.style.height = `${ch}px`
      canvasSizeRef.current = { w: cw, h: ch, dpr }
    }

    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, cw, ch)

    const { scale, ox, oy } = containRect(naturalW, naturalH, cw, ch)
    ctx.lineWidth = Math.max(2, Math.round(Math.min(cw, ch) / 180))

    for (const tr of tracksRef.current.values()) {
      const [x1, y1, x2, y2] = tr.box
      const bw = (x2 - x1) * scale
      const bh = (y2 - y1) * scale
      if (bw < 2 || bh < 2) continue
      ctx.strokeStyle = tr.color
      ctx.strokeRect(ox + x1 * scale, oy + y1 * scale, bw, bh)
    }
  }, [])

  const mergeDetections = useCallback((dets, naturalW, naturalH) => {
    frameSizeRef.current = { w: naturalW, h: naturalH }
    const now = performance.now()
    const tracks = tracksRef.current
    const used = new Set()

    const drawable = dets.filter(
      (d) => d.box && d.box.length >= 4 && d.box[2] - d.box[0] > 2 && d.box[3] - d.box[1] > 2,
    )

    for (const d of drawable) {
      const color = boxColor(d)
      let key =
        d.track_id != null && d.track_id !== undefined
          ? `t-${d.track_id}`
          : null

      if (!key) {
        let bestId = null
        let bestIou = IOU_MATCH
        for (const [id, tr] of tracks) {
          if (used.has(id)) continue
          const iou = boxIou(tr.target || tr.box, d.box)
          if (iou > bestIou) {
            bestIou = iou
            bestId = id
          }
        }
        key = bestId || `tmp-${tempIdRef.current++}`
      }

      used.add(key)
      const prev = tracks.get(key)
      if (prev) {
        prev.target = d.box
        prev.color = color
        prev.lastSeen = now
        prev.meta = d
      } else {
        tracks.set(key, {
          box: [...d.box],
          target: [...d.box],
          color,
          lastSeen: now,
          meta: d,
        })
      }
    }

    lastDrawRef.current = { dets: drawable, w: naturalW, h: naturalH }
  }, [])

  const drawStillDetections = useCallback(
    (dets, naturalW, naturalH) => {
      tracksRef.current.clear()
      mergeDetections(dets, naturalW, naturalH)
      // Snap boxes to target (no lerp for stills)
      for (const tr of tracksRef.current.values()) {
        tr.box = [...tr.target]
      }
      paintTracks()
    },
    [mergeDetections, paintTracks],
  )

  const sendFrame = useCallback(async (blob, naturalW, naturalH, sentW, sentH) => {
    const enabledMap = enabledRef.current
    const list = MODELS.filter((m) => enabledMap[m.id]).map((m) => m.id)
    if (!blob || list.length === 0 || busyRef.current) return
    busyRef.current = true
    const mode = sourceModeRef.current
    try {
      const sessionId = mode === 'image' ? 'still' : sessionIdRef.current
      const form = new FormData()
      form.append('image', blob, 'frame.jpg')
      form.append('session_id', sessionId)
      form.append('enabled', JSON.stringify(list))
      form.append('correct_direction', directionRef.current)
      const res = await fetch('/api/detect', { method: 'POST', body: form })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `HTTP ${res.status}`)
      }
      const data = await res.json()

      const sx = naturalW && sentW ? naturalW / sentW : 1
      const sy = naturalH && sentH ? naturalH / sentH : 1
      const dets = (data.detections ?? []).map((d) => {
        if (!d.box || d.box.length < 4) return d
        const [x1, y1, x2, y2] = d.box
        return { ...d, box: [x1 * sx, y1 * sy, x2 * sx, y2 * sy] }
      })

      const nw = naturalW || sentW || 640
      const nh = naturalH || sentH || 480
      if (mode === 'image') {
        drawStillDetections(dets, nw, nh)
      } else {
        mergeDetections(dets, nw, nh)
      }

      startTransition(() => {
        setPeopleCount(data.people_count ?? 0)
        setAlerts(data.alerts ?? [])
        setDetections(data.detections ?? [])
        if (mode !== 'image') setStatus('مباشر')
        else setStatus('تم')
      })
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
  }, [drawStillDetections, mergeDetections])

  const captureFromVideo = useCallback(() => {
    if (busyRef.current) return
    const video = videoRef.current
    if (!video || video.readyState < 2) return
    const vw = video.videoWidth
    const vh = video.videoHeight
    if (!vw || !vh) return

    const maxW = 640
    const scale = Math.min(1, maxW / vw)
    const w = Math.max(1, Math.round(vw * scale))
    const h = Math.max(1, Math.round(vh * scale))

    let canvas = captureCanvasRef.current
    if (!canvas) {
      canvas = document.createElement('canvas')
      captureCanvasRef.current = canvas
    }
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w
      canvas.height = h
    }
    const ctx = canvas.getContext('2d', { alpha: false })
    ctx.drawImage(video, 0, 0, w, h)
    canvas.toBlob(
      (blob) => {
        if (blob) sendFrame(blob, vw, vh, w, h)
      },
      'image/jpeg',
      0.6,
    )
  }, [sendFrame])

  const captureFromImage = useCallback(() => {
    const img = imageRef.current
    if (!img || !img.complete) return
    const nw = img.naturalWidth
    const nh = img.naturalHeight
    const maxW = 1280
    const scale = Math.min(1, maxW / nw)
    const w = Math.max(1, Math.round(nw * scale))
    const h = Math.max(1, Math.round(nh * scale))
    const canvas = document.createElement('canvas')
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d', { alpha: false })
    ctx.drawImage(img, 0, 0, w, h)
    canvas.toBlob(
      (blob) => {
        if (blob) sendFrame(blob, nw, nh, w, h)
      },
      'image/jpeg',
      0.85,
    )
  }, [sendFrame])

  // Live: detect + continuous smooth box painting
  useEffect(() => {
    if (!running || !enabledKey) return
    if (sourceMode !== 'webcam' && sourceMode !== 'video') return

    const hasHeavy = enabledKey.includes('dmcount')
    const minGapMs = hasHeavy ? 2500 : 350
    let alive = true
    let lastTry = 0
    let raf = 0

    const tick = (t) => {
      if (!alive) return

      const tracks = tracksRef.current
      const now = t
      for (const [id, tr] of [...tracks.entries()]) {
        if (now - tr.lastSeen > TRACK_HOLD_MS) {
          tracks.delete(id)
          continue
        }
        tr.box = lerpBox(tr.box, tr.target, LERP_SPEED)
      }
      paintTracks()

      if (t - lastTry >= minGapMs && !busyRef.current) {
        lastTry = t
        captureFromVideo()
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => {
      alive = false
      cancelAnimationFrame(raf)
    }
  }, [running, enabledKey, sourceMode, captureFromVideo, paintTracks])

  // Analyze still image when ready / models change
  useEffect(() => {
    if (sourceMode !== 'image' || !running || enabledList.length === 0) return
    captureFromImage()
  }, [sourceMode, running, enabledList.join(','), captureFromImage])

  const startWebcam = async () => {
    setError('')
    stopCamera()
    try {
      // Ask permission first so device labels are available
      const warm = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      })
      warm.getTracks().forEach((t) => t.stop())

      const devices = await navigator.mediaDevices.enumerateDevices()
      const cameras = devices.filter((d) => d.kind === 'videoinput')
      // Prefer real laptop/webcam over Iriun / virtual cams
      const preferred =
        cameras.find(
          (d) =>
            d.label &&
            !/iriun|obs|virtual|droidcam|epoccam|manyCam/i.test(d.label),
        ) || cameras[0]

      const videoBase = {
        frameRate: { ideal: 60, max: 60 },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      }
      const constraints = preferred?.deviceId
        ? {
            video: { ...videoBase, deviceId: { exact: preferred.deviceId } },
            audio: false,
          }
        : { video: videoBase, audio: false }

      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      const track = stream.getVideoTracks()[0]
      const settings = track?.getSettings?.() || {}
      const fps = settings.frameRate ? Math.round(settings.frameRate) : null
      setSourceMode('webcam')
      setRunning(true)
      sessionIdRef.current = createSessionId()
      setStatus(
        preferred?.label
          ? `مباشر · ${preferred.label}${fps ? ` · ${fps}fps` : ''}`
          : fps
            ? `مباشر · ${fps}fps`
            : 'مباشر',
      )
      clearTracks()
    } catch (err) {
      setError(
        'تعذر فتح الكاميرا. اسمح للموقع بالكاميرا من قفل المتصفح، أو استخدم «رفع صورة / فيديو». ' +
          (err.message || err),
      )
    }
  }

  const onFileChange = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    stopCamera()
    clearTracks()
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
      sessionIdRef.current = createSessionId()
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
    clearTracks()
    setStatus('متوقف')
  }

  useEffect(() => () => stopCamera(), [stopCamera])

  useEffect(() => {
    const onResize = () => {
      if (frameSizeRef.current.w) paintTracks()
    }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [paintTracks])

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
            <h3>اتجاه المشي الصحيح</h3>
            <p className="direction-hint">
              يتتبع الأشخاص: أخضر = صح · أحمر = غلط · أصفر = يتتبع
            </p>
            <div className="direction-grid">
              {DIRECTIONS.map((d) => (
                <button
                  key={d.id}
                  type="button"
                  className={`direction-btn ${correctDirection === d.id ? 'on' : ''}`}
                  onClick={() => changeDirection(d.id)}
                >
                  <span className="direction-arrow">{d.arrow}</span>
                  <span className="direction-label">{d.label}</span>
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
        <div className="stage" ref={stageRef}>
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
                    {a === 'wrong_way' && 'شخص عكس الاتجاه'}
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
                    style={{ background: boxColor(d) }}
                  />
                  {displayLabel(d)} — {(d.score * 100).toFixed(0)}%
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

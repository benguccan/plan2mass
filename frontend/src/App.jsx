import { useEffect, useMemo, useRef, useState } from "react"
import * as THREE from "three"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls"
import { GLTFExporter } from "three/examples/jsm/exporters/GLTFExporter.js"
import { RGBELoader } from "three/examples/jsm/loaders/RGBELoader.js"
import {
  buildWallGraph,
  classifyOpenings,
  matchOpeningsToWalls,
  splitWallsByOpenings,
} from "./wallGraph"
import {
  createBabylonViewer,
  disposeBabylonViewer,
} from "./babylonRenderer"
import { buildWorldTransform } from "./worldTransform"
import {
  AUTH_GATE_OPEN_APP_MESSAGE,
  AUTH_GATE_ROUTE_MESSAGE,
  canAccessApp,
  getProtectedButtonStyle,
} from "./authGate"

const API = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
const AUTH_STORAGE_KEY = "plan2mass:auth-token"
const LOCAL_AUTH_USERS_KEY = "plan2mass:local-auth-users"

const OUTER_WALL_THICKNESS = 0.30
const INNER_WALL_THICKNESS = 0.12
const SLAB_THICKNESS = 0.18
const ROOF_THICKNESS = 0.18

const WINDOW_SILL_HEIGHT = 0.98
const WINDOW_HEIGHT = 0.94
const DOOR_HEIGHT_RATIO = 0.68
const BUILDING_FLOOR_GAP = 0.06

const DOOR_OPENING_WIDTH_PX = 92
const WINDOW_OPENING_WIDTH_PX = 120

const TARGET_MODEL_SIZE = 22
const LAST_PROJECT_STORAGE_KEY = "plan2mass:last-project-id"
const defaultPalette = {
  walls: "#ddd4c7",
  slabs: "#d2c0aa",
  roof: "#bfc5cb",
  frames: "#2f363e",
  glass: "#a7bac9",
  door: "#845535",
  ground: "#cec5b5",
}

function createLocalDemoProject(floorHeight = 3.2) {
  const basePolygon = [
    [40, 40],
    [540, 40],
    [540, 300],
    [40, 300],
  ]
  const floor1CleanGeometry = {
    polygon: basePolygon,
    stairs: [
      {
        id: "f1-stair-1",
        bounds: [228, 52, 312, 88],
        direction: "down",
        steps: 5,
      },
    ],
    rooms: [
      { id: "f1c-r1", x: 105, y: 172 },
      { id: "f1c-r2", x: 280, y: 95 },
      { id: "f1c-r3", x: 455, y: 95 },
      { id: "f1c-r4", x: 280, y: 222 },
      { id: "f1c-r5", x: 455, y: 222 },
    ],
    inner_walls: [
      [180, 40, 180, 300],
      [360, 40, 360, 300],
      [180, 155, 540, 155],
    ],
    doors: [
      { x: 180, y: 248, width: 88 },
      { x: 270, y: 155, width: 92 },
      { x: 360, y: 110, width: 84 },
      { x: 360, y: 210, width: 84 },
      { x: 300, y: 300, width: 104 },
    ],
    windows: [
      { x: 105, y: 40, width: 86 },
      { x: 270, y: 40, width: 88 },
      { x: 460, y: 40, width: 88 },
      { x: 40, y: 170, width: 86 },
      { x: 115, y: 300, width: 88 },
      { x: 540, y: 225, width: 86 },
    ],
  }
  const floor2CleanGeometry = {
    polygon: basePolygon,
    stairs: [
      {
        id: "f2-stair-1",
        bounds: [238, 52, 322, 90],
        direction: "down",
        steps: 5,
      },
    ],
    rooms: [
      { id: "f2c-r1", x: 105, y: 170 },
      { id: "f2c-r2", x: 270, y: 95 },
      { id: "f2c-r3", x: 450, y: 95 },
      { id: "f2c-r4", x: 270, y: 232 },
      { id: "f2c-r5", x: 450, y: 232 },
    ],
    inner_walls: [
      [180, 40, 180, 195],
      [180, 195, 230, 195],
      [230, 195, 230, 300],
      [360, 40, 360, 300],
      [230, 195, 360, 195],
      [360, 155, 540, 155],
    ],
    doors: [
      { x: 180, y: 120, width: 88 },
      { x: 270, y: 195, width: 92 },
      { x: 360, y: 110, width: 84 },
      { x: 360, y: 225, width: 84 },
    ],
    windows: [
      { x: 105, y: 40, width: 86 },
      { x: 270, y: 40, width: 88 },
      { x: 455, y: 40, width: 88 },
      { x: 40, y: 170, width: 86 },
      { x: 540, y: 225, width: 86 },
      { x: 110, y: 300, width: 88 },
      { x: 300, y: 300, width: 88 },
    ],
  }
  const floor3CleanGeometry = {
    polygon: basePolygon,
    stairs: [
      {
        id: "f3-stair-1",
        bounds: [223, 52, 307, 88],
        direction: "down",
        steps: 5,
      },
    ],
    rooms: [
      { id: "f3c-r1", x: 105, y: 100 },
      { id: "f3c-r2", x: 105, y: 232 },
      { id: "f3c-r3", x: 272, y: 178 },
      { id: "f3c-r4", x: 452, y: 100 },
      { id: "f3c-r5", x: 452, y: 232 },
    ],
    inner_walls: [
      [160, 40, 160, 125],
      [40, 125, 160, 125],
      [100, 125, 100, 195],
      [40, 195, 100, 195],
      [100, 195, 260, 195],
      [260, 145, 260, 300],
      [360, 40, 360, 155],
      [260, 155, 540, 155],
    ],
    doors: [
      { x: 100, y: 162, width: 52 },
      { x: 132, y: 125, width: 40 },
      { x: 205, y: 195, width: 92 },
      { x: 300, y: 155, width: 92 },
      { x: 360, y: 98, width: 84 },
    ],
    windows: [
      { x: 105, y: 40, width: 86 },
      { x: 245, y: 40, width: 88 },
      { x: 455, y: 40, width: 88 },
      { x: 40, y: 80, width: 84 },
      { x: 40, y: 220, width: 84 },
      { x: 185, y: 300, width: 88 },
      { x: 420, y: 300, width: 88 },
      { x: 540, y: 85, width: 84 },
      { x: 540, y: 225, width: 84 },
    ],
  }

  const floorTemplates = [
    {
      rooms: floor1CleanGeometry.rooms,
      inner_walls: floor1CleanGeometry.inner_walls,
      doors: floor1CleanGeometry.doors,
      windows: floor1CleanGeometry.windows,
    },
    {
      rooms: floor2CleanGeometry.rooms,
      inner_walls: floor2CleanGeometry.inner_walls,
      doors: floor2CleanGeometry.doors,
      windows: floor2CleanGeometry.windows,
    },
    {
      rooms: floor3CleanGeometry.rooms,
      inner_walls: floor3CleanGeometry.inner_walls,
      doors: floor3CleanGeometry.doors,
      windows: floor3CleanGeometry.windows,
    },
  ]

  const makeFloor = (index) => ({
    floor_index: index + 1,
    image_url:
      index === 0
        ? "/demo/floor_1_clean.png"
        : index === 1
          ? "/demo/floor_2_clean.png"
          : index === 2
            ? "/demo/floor_3_clean.png"
            : `/demo/floor_${index + 1}.png`,
    polygon:
      index === 0
        ? floor1CleanGeometry.polygon
        : index === 1
          ? floor2CleanGeometry.polygon
          : index === 2
            ? floor3CleanGeometry.polygon
            : basePolygon,
    inner_walls: floorTemplates[index].inner_walls,
    doors: floorTemplates[index].doors,
    windows: floorTemplates[index].windows,
    stairs:
      index === 0
        ? floor1CleanGeometry.stairs
        : index === 1
          ? floor2CleanGeometry.stairs
          : index === 2
            ? floor3CleanGeometry.stairs
            : [],
    rooms: floorTemplates[index].rooms,
    summary: {
      polygon_points:
        (
          index === 0
            ? floor1CleanGeometry.polygon
            : index === 1
              ? floor2CleanGeometry.polygon
              : index === 2
                ? floor3CleanGeometry.polygon
                : basePolygon
        )
          .length,
      inner_wall_count: floorTemplates[index].inner_walls.length,
      door_count: floorTemplates[index].doors.length,
      window_count: floorTemplates[index].windows.length,
      room_count: floorTemplates[index].rooms.length,
    },
    debug: {},
    height: floorHeight,
  })

  const floors = [makeFloor(1), makeFloor(0), makeFloor(2)]

  return {
    project_id: "demo-local",
    floor_count: floors.length,
    floor_height: floorHeight,
    building_height: Number((floors.length * floorHeight).toFixed(1)),
    summary: {
      room_count: 12,
      door_count: 9,
      window_count: 21,
      inner_wall_count: 12,
    },
    floors,
  }
}

function buildLocalPlanSvgDataUrl(floor) {
  if (!floor?.polygon?.length) return null

  const points = floor.polygon
  const xs = points.map(([x]) => x)
  const ys = points.map(([, y]) => y)
  const minX = Math.min(...xs)
  const minY = Math.min(...ys)
  const width = Math.max(...xs) - minX
  const height = Math.max(...ys) - minY
  const pad = 28
  const viewWidth = width + pad * 2
  const viewHeight = height + pad * 2

  const mapPoint = (x, y) => `${(x - minX + pad).toFixed(1)},${(y - minY + pad).toFixed(1)}`
  const polygonPoints = points.map(([x, y]) => mapPoint(x, y)).join(" ")

  const walls = (floor.inner_walls || [])
    .map(([x1, y1, x2, y2]) => {
      return `<line x1="${(x1 - minX + pad).toFixed(1)}" y1="${(y1 - minY + pad).toFixed(1)}" x2="${(x2 - minX + pad).toFixed(1)}" y2="${(y2 - minY + pad).toFixed(1)}" stroke="#2a2f35" stroke-width="8" stroke-linecap="square" />`
    })
    .join("")

  const doors = (floor.doors || [])
    .map((door) => {
      const cx = (door.x - minX + pad).toFixed(1)
      const cy = (door.y - minY + pad).toFixed(1)
      return `<circle cx="${cx}" cy="${cy}" r="8" fill="#8a4f2a" opacity="0.9" />`
    })
    .join("")

  const windows = (floor.windows || [])
    .map((window) => {
      const cx = window.x - minX + pad
      const cy = window.y - minY + pad
      return `<rect x="${(cx - 14).toFixed(1)}" y="${(cy - 4).toFixed(1)}" width="28" height="8" rx="3" fill="#b9d8e8" opacity="0.95" />`
    })
    .join("")

  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="${viewWidth}" height="${viewHeight}" viewBox="0 0 ${viewWidth} ${viewHeight}">
      <rect width="100%" height="100%" fill="#f8f7f4" />
      <polygon points="${polygonPoints}" fill="#ffffff" stroke="#22272d" stroke-width="10" stroke-linejoin="round" />
      ${walls}
      ${windows}
      ${doors}
    </svg>
  `

  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
}

function App() {
  const canvasRef = useRef(null)
  const viewerPanelRef = useRef(null)
  const modelRef = useRef(null)
  const sceneRef = useRef(null)
  const cameraRef = useRef(null)
  const controlsRef = useRef(null)
  const animationRef = useRef(null)
  const viewerRef = useRef(null)

  const [project, setProject] = useState(null)
  const [routePath, setRoutePath] = useState(() =>
    window.location.pathname === "/app" ? "/app" : "/"
  )
  const [activeFloor, setActiveFloor] = useState("building")
  const [files, setFiles] = useState([])
  const [projectId, setProjectId] = useState(null)
  const [selectedFloorCount, setSelectedFloorCount] = useState(3)
  const [floorHeight, setFloorHeight] = useState(3.2)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [loadingPhaseIndex, setLoadingPhaseIndex] = useState(0)
  const [toasts, setToasts] = useState([])
  const [isDropActive, setIsDropActive] = useState(false)
  const [viewportWidth, setViewportWidth] = useState(() => window.innerWidth)
  const [showGeometryDebug, setShowGeometryDebug] = useState(false)
  const [authToken, setAuthToken] = useState(() => window.localStorage.getItem(AUTH_STORAGE_KEY) || "")
  const [authUser, setAuthUser] = useState(null)
  const [authMode, setAuthMode] = useState("login")
  const [authName, setAuthName] = useState("")
  const [authEmail, setAuthEmail] = useState("")
  const [authPassword, setAuthPassword] = useState("")
  const [authPasswordConfirm, setAuthPasswordConfirm] = useState("")
  const [authError, setAuthError] = useState("")
  const [authSuccess, setAuthSuccess] = useState("")
  const [isAuthLoading, setIsAuthLoading] = useState(false)
  const [isAuthResolved, setIsAuthResolved] = useState(() => !window.localStorage.getItem(AUTH_STORAGE_KEY))
  const [userProjects, setUserProjects] = useState([])
  const [colorPaletteApplied, setColorPaletteApplied] = useState(defaultPalette)
  const [colorPaletteDraft, setColorPaletteDraft] = useState(defaultPalette)
  const [sidebarSections, setSidebarSections] = useState({
    upload: true,
    settings: true,
    floors: true,
  })
  const didInitProjectRef = useRef(false)
  const authGateToastShownRef = useRef(false)
  const localDemoProject = useMemo(() => createLocalDemoProject(floorHeight), [floorHeight])
  const canEnterApp = canAccessApp(authToken, authUser)

  const authValidation = useMemo(() => {
    const email = authEmail.trim().toLowerCase()
    const displayName = authName.trim()
    const issues = []
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

    if (!emailRegex.test(email)) {
      issues.push("Please enter a valid email address.")
    }

    if (authPassword.length < 8) {
      issues.push("Password must be at least 8 characters.")
    }

    if (authMode === "register") {
      if (displayName.length < 2) {
        issues.push("Display name must be at least 2 characters.")
      }
      if (authPassword !== authPasswordConfirm) {
        issues.push("Password confirmation does not match.")
      }
    }

    return {
      email,
      displayName,
      issues,
      isValid: issues.length === 0,
    }
  }, [authEmail, authName, authPassword, authPasswordConfirm, authMode])

  const loadingPhases = [
    "Analyzing plan...",
    "Detecting walls...",
    "Estimating rooms...",
    "Generating 3D model...",
  ]

  const applyLocalDemoProject = () => {
    setProject(localDemoProject)
    setProjectId("demo-local")
    setError("")
  }

  const pushToast = (message, tone = "neutral") => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`
    setToasts((current) => [...current, { id, message, tone }])
    window.setTimeout(() => {
      setToasts((current) => current.filter((item) => item.id !== id))
    }, 2600)
  }

  const authHeaders = authToken ? { Authorization: `Bearer ${authToken}` } : {}

  const fetchAuthMe = async (token = authToken) => {
    if (!token) {
      setAuthUser(null)
      return null
    }
    const res = await fetch(`${API}/auth/me`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
    if (!res.ok) {
      setAuthUser(null)
      return null
    }
    const json = await res.json()
    setAuthUser(json.user || null)
    return json.user || null
  }

  const fetchUserProjects = async (token = authToken) => {
    if (!token) {
      setUserProjects([])
      return
    }
    const res = await fetch(`${API}/users/me/projects`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
    if (!res.ok) {
      return
    }
    const json = await res.json()
    setUserProjects(json.projects || [])
  }

  const persistLogin = (token, user) => {
    setAuthToken(token)
    setAuthUser(user || null)
    window.localStorage.setItem(AUTH_STORAGE_KEY, token)
  }

  const logout = () => {
    setAuthToken("")
    setAuthUser(null)
    setUserProjects([])
    window.localStorage.removeItem(AUTH_STORAGE_KEY)
  }

  const navigateTo = (path) => {
    if (path === "/app" && !canEnterApp) {
      setAuthMode("login")
      setAuthError(AUTH_GATE_OPEN_APP_MESSAGE)
      if (!authGateToastShownRef.current) {
        pushToast("Open App icin once hesabiniza giris yapin.", "error")
        authGateToastShownRef.current = true
      }
      path = "/"
    }

    if (window.location.pathname !== path) {
      window.history.pushState({}, "", path)
    }
    setRoutePath(path)
  }

  const toggleSidebarSection = (section) => {
    setSidebarSections((current) => ({
      ...current,
      [section]: !current[section],
    }))
  }

  const applyPalettePreset = (preset) => {
    if (preset === "warm") {
      const next = {
        walls: "#ddd4c7",
        slabs: "#d2c0aa",
        roof: "#bfc5cb",
        frames: "#2f363e",
        glass: "#a7bac9",
        door: "#845535",
        ground: "#cec5b5",
      }
      setColorPaletteDraft(next)
      setColorPaletteApplied(next)
      return
    }
    if (preset === "modern") {
      const next = {
        walls: "#d8d8d4",
        slabs: "#b5b1a9",
        roof: "#a5adb5",
        frames: "#232930",
        glass: "#8fa8bb",
        door: "#6e4a35",
        ground: "#c8c7c2",
      }
      setColorPaletteDraft(next)
      setColorPaletteApplied(next)
      return
    }
    const next = {
      walls: "#ece4d9",
      slabs: "#cdb9a2",
      roof: "#d4d9dd",
      frames: "#3f4952",
      glass: "#b4c7d6",
      door: "#9a6a48",
      ground: "#d7cfbf",
    }
    setColorPaletteDraft(next)
    setColorPaletteApplied(next)
  }

  useEffect(() => {
    const onPopState = () => setRoutePath(window.location.pathname === "/app" ? "/app" : "/")
    window.addEventListener("popstate", onPopState)
    return () => window.removeEventListener("popstate", onPopState)
  }, [])

  useEffect(() => {
    let isActive = true

    const resolveAuth = async () => {
      if (!authToken) {
        if (!isActive) return
        setAuthUser(null)
        setUserProjects([])
        setIsAuthResolved(true)
        return
      }

      if (!isActive) return
      setIsAuthResolved(false)

      const user = await fetchAuthMe(authToken)
      if (!isActive) return

      if (!user) {
        window.localStorage.removeItem(AUTH_STORAGE_KEY)
        setAuthToken("")
        setAuthUser(null)
        setUserProjects([])
        setIsAuthResolved(true)
        return
      }

      await fetchUserProjects(authToken)
      if (!isActive) return
      setIsAuthResolved(true)
    }

    resolveAuth()
    return () => {
      isActive = false
    }
  }, [authToken])

  useEffect(() => {
    if (canEnterApp) {
      authGateToastShownRef.current = false
    }
  }, [canEnterApp])

  useEffect(() => {
    if (!isAuthResolved) return
    if (routePath !== "/app") return
    if (canEnterApp) return

    setAuthMode("login")
    setAuthError(AUTH_GATE_ROUTE_MESSAGE)
    if (!authGateToastShownRef.current) {
      pushToast("Once hesap olusturup giris yapmaniz gerekiyor.", "error")
      authGateToastShownRef.current = true
    }

    if (window.location.pathname !== "/") {
      window.history.replaceState({}, "", "/")
    }
    setRoutePath("/")
  }, [routePath, canEnterApp, isAuthResolved])

  useEffect(() => {
    const onResize = () => setViewportWidth(window.innerWidth)
    window.addEventListener("resize", onResize)
    return () => window.removeEventListener("resize", onResize)
  }, [])

  useEffect(() => {
    if (!isLoading) {
      setLoadingPhaseIndex(0)
      return
    }

    const timer = window.setInterval(() => {
      setLoadingPhaseIndex((value) => (value + 1) % loadingPhases.length)
    }, 1400)

    return () => window.clearInterval(timer)
  }, [isLoading])

  useEffect(() => {
    if (didInitProjectRef.current) return
    didInitProjectRef.current = true

    const lastProjectId = window.localStorage.getItem(LAST_PROJECT_STORAGE_KEY)
    if (lastProjectId && lastProjectId !== "demo-local" && lastProjectId !== "demo") {
      loadProject(lastProjectId)
      return
    }

    setIsLoading(true)
    applyLocalDemoProject()
    setIsLoading(false)
  }, [])

  const loadProject = async (id) => {
    setIsLoading(true)
    setError("")

    try {
      if (id === "demo" || id === "demo-local") {
        applyLocalDemoProject()
        return
      }

      const res = await fetch(`${API}/projects/${id}?floor_height=${floorHeight}`)
      const json = await res.json()

      if (!res.ok) {
        throw new Error(json?.detail || "Project could not be loaded")
      }

      console.log("[loadProject] backend response", {
        project_id: json?.project_id || id,
        floors_length: json?.floors?.length || 0,
        floor_0_polygon_length: json?.floors?.[0]?.polygon?.length || 0,
        floor_0_inner_walls_length: json?.floors?.[0]?.inner_walls?.length || 0,
        floor_0_doors_length: json?.floors?.[0]?.doors?.length || 0,
        floor_0_windows_length: json?.floors?.[0]?.windows?.length || 0,
        floor_0_image_url: json?.floors?.[0]?.image_url || null,
      })

      setProject(json)
      setProjectId(id)
      setActiveFloor((json?.floors?.length || 0) <= 1 ? 0 : "building")
      if (id && id !== "demo" && id !== "demo-local") {
        window.localStorage.setItem(LAST_PROJECT_STORAGE_KEY, id)
      }

      if (typeof json.floor_height === "number") {
        setFloorHeight(json.floor_height)
      }
    } catch (e) {
      const isNetworkError =
        String(e?.message || "").toLowerCase().includes("failed to fetch") ||
        String(e?.message || "").toLowerCase().includes("networkerror")

      if (id === "demo" || id === "demo-local") {
        applyLocalDemoProject()
        pushToast("Local demo mode enabled", "success")
      } else if (isNetworkError) {
        // Backend is unavailable: fall back to local demo instead of blocking the app.
        window.localStorage.removeItem(LAST_PROJECT_STORAGE_KEY)
        applyLocalDemoProject()
        pushToast("Backend offline, local demo opened", "error")
      } else {
        setError(e.message || "Project could not be loaded")
      }
    } finally {
      setIsLoading(false)
    }
  }

  const getLocalAuthUsers = () => {
    try {
      const raw = window.localStorage.getItem(LOCAL_AUTH_USERS_KEY)
      const parsed = raw ? JSON.parse(raw) : []
      return Array.isArray(parsed) ? parsed : []
    } catch {
      return []
    }
  }

  const setLocalAuthUsers = (users) => {
    window.localStorage.setItem(LOCAL_AUTH_USERS_KEY, JSON.stringify(users))
  }

  const uploadPlan = async () => {
    if (files.length === 0) return

    const selectedPdfCount = files.filter((file) => file.name.toLowerCase().endsWith(".pdf")).length
    const selectedImageCount = files.length - selectedPdfCount

    if (files.length > 3 && selectedPdfCount === 0) {
      const message = "You can upload up to 3 floor images."
      setError(message)
      pushToast(message, "error")
      return
    }

    if (selectedImageCount > 0 && selectedPdfCount === 0 && selectedImageCount !== selectedFloorCount) {
      const message = `Selected floor count is ${selectedFloorCount}, but ${selectedImageCount} image file(s) were added.`
      setError(message)
      pushToast(message, "error")
      return
    }

    setIsLoading(true)
    setError("")

    try {
      const fd = new FormData()
      files.forEach((file) => fd.append("files", file))
      fd.append("floor_count", String(selectedFloorCount))

      const res = await fetch(`${API}/projects/upload`, {
        method: "POST",
        headers: {
          ...authHeaders,
        },
        body: fd,
      })

      const json = await res.json()

      if (!res.ok) {
        throw new Error(json?.detail || "Upload failed")
      }

      if (typeof json?.floor_count === "number" && json.floor_count !== selectedFloorCount) {
        throw new Error(
          `Expected ${selectedFloorCount} floor(s), but the uploaded plan produced ${json.floor_count} floor(s).`
        )
      }

      setProjectId(json.project_id)
      await loadProject(json.project_id)
      await fetchUserProjects()
      pushToast("Plan uploaded", "success")
    } catch (e) {
      setError(e.message || "Upload failed")
      pushToast(e.message || "Upload failed", "error")
    } finally {
      setIsLoading(false)
    }
  }

  const submitAuth = async () => {
    setIsAuthLoading(true)
    setAuthError("")
    setAuthSuccess("")
    try {
      if (!authValidation.isValid) {
        throw new Error(authValidation.issues[0] || "Please check the form fields.")
      }

      const endpoint = authMode === "register" ? "/auth/register" : "/auth/login"
      const payload = {
        email: authValidation.email,
        password: authPassword,
      }
      if (authMode === "register") {
        payload.display_name = authValidation.displayName
      }
      const res = await fetch(`${API}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      })
      const json = await res.json()
      if (!res.ok) {
        throw new Error(json?.detail || "Authentication failed")
      }

      persistLogin(json.token, json.user)
      await fetchUserProjects(json.token)
      setAuthPassword("")
      setAuthPasswordConfirm("")
      setAuthError("")

      if (authMode === "register") {
        setAuthSuccess("Account created successfully. Redirecting to the app...")
        pushToast("Account created and signed in", "success")
      } else {
        setAuthSuccess("Signed in successfully. Redirecting to the app...")
        pushToast("Signed in", "success")
      }
      navigateTo("/app")
    } catch (e) {
      const msg = String(e?.message || "")
      if (msg.toLowerCase().includes("failed to fetch")) {
        // Presentation-safe fallback: auth still works in local mode when backend is down.
        const users = getLocalAuthUsers()
        if (authMode === "register") {
          const exists = users.find((u) => u.email === authValidation.email)
          if (exists) {
            setAuthError("This email is already registered in local mode.")
          } else {
            const user = {
              id: Date.now(),
              email: authValidation.email,
              display_name: authValidation.displayName || "User",
              password: authPassword,
            }
            users.push(user)
            setLocalAuthUsers(users)
            persistLogin(`local-${user.id}`, {
              id: user.id,
              email: user.email,
              display_name: user.display_name,
            })
            setAuthSuccess("Local account created. Redirecting to the app...")
            pushToast("Signed in (local mode)", "success")
            navigateTo("/app")
          }
        } else {
          const match = users.find((u) => u.email === authValidation.email && u.password === authPassword)
          if (!match) {
            setAuthError("Backend offline and no matching local account found.")
          } else {
            persistLogin(`local-${match.id}`, {
              id: match.id,
              email: match.email,
              display_name: match.display_name || "User",
            })
            setAuthSuccess("Signed in (local mode). Redirecting to the app...")
            pushToast("Signed in (local mode)", "success")
            navigateTo("/app")
          }
        }
      } else {
        setAuthError(msg || "Authentication failed")
      }
    } finally {
      setIsAuthLoading(false)
    }
  }

  const refreshAnalysis = async () => {
    if (!projectId) return
    await loadProject(projectId)
    pushToast("Analysis refreshed", "success")
  }

  const resetToDemoProject = async () => {
    setFiles([])
    setError("")
    setSelectedFloorCount(3)
    setActiveFloor("building")
    window.localStorage.removeItem(LAST_PROJECT_STORAGE_KEY)
    applyLocalDemoProject()
    pushToast("Returned to demo project", "success")
  }

  const downloadModel = () => {
    if (!viewerRef.current) return

    viewerRef.current
      .exportGlb(`${project?.project_id || "plan2mass"}-model`)
      .then(() => {
        pushToast("3D model exported as GLB", "success")
      })
      .catch((err) => {
        console.error("3D GLB export error:", err)
        pushToast("3D model export failed", "error")
      })
  }

  const floors = project?.floors || []
  const isLocalDemoProject = project?.project_id === "demo-local"
  const resolvedActiveFloor =
    activeFloor === "building"
      ? "building"
      : isLocalDemoProject
        ? [1, 0, 2][activeFloor] ?? activeFloor
        : activeFloor
  const activeFloorData =
    resolvedActiveFloor === "building"
      ? floors[0] || null
      : floors[resolvedActiveFloor] || floors[0] || null

  const geometryMeta = useMemo(() => {
    return buildProjectGeometryMeta(project, TARGET_MODEL_SIZE)
  }, [project])
  const viewerProject = useMemo(() => {
    if (!project) return project
    if (project.project_id !== "demo-local") return project
    if (activeFloor !== "building") return project

    return {
      ...project,
      floors: [...(project.floors || [])].sort((a, b) => (a?.floor_index || 0) - (b?.floor_index || 0)),
    }
  }, [project, activeFloor])

  const activePlanDebug = useMemo(() => {
    if (!activeFloorData?.polygon?.length) return null

    const graph = buildWallGraph({
      polygon: activeFloorData.polygon || [],
      innerWalls: activeFloorData.inner_walls || [],
    })

    const classified = classifyOpenings({
      graph,
      doors: activeFloorData.doors || [],
      windows: activeFloorData.windows || [],
      doorWidthPx: DOOR_OPENING_WIDTH_PX,
      windowWidthPx: WINDOW_OPENING_WIDTH_PX,
    })

    const { matched } = matchOpeningsToWalls({
      graph,
      openings: classified,
    })

    const points = []
    ;(activeFloorData.polygon || []).forEach(([x, y]) => points.push({ x, y }))
    graph.walls.forEach((wall) => {
      points.push({ x: wall.line[0], y: wall.line[1] })
      points.push({ x: wall.line[2], y: wall.line[3] })
    })
    matched.forEach((opening) => points.push(opening.point))

    if (!points.length) return null

    const minX = Math.min(...points.map((p) => p.x))
    const maxX = Math.max(...points.map((p) => p.x))
    const minY = Math.min(...points.map((p) => p.y))
    const maxY = Math.max(...points.map((p) => p.y))
    const pad = 24
    const width = Math.max(1, maxX - minX)
    const height = Math.max(1, maxY - minY)

    const mapPoint = (x, y) => ({
      x: ((x - minX) / width) * (1000 - pad * 2) + pad,
      y: ((y - minY) / height) * (540 - pad * 2) + pad,
    })

    return {
      graph,
      matched,
      mapPoint,
      viewBox: "0 0 1000 540",
    }
  }, [activeFloorData])

  const currentPlanImage = useMemo(() => {
    if (!project?.floors?.length) return null
    if (activeFloor === "building") return project.floors[0]?.image_url || null
    return project.floors[resolvedActiveFloor]?.image_url || project.floors[0]?.image_url || null
  }, [project, activeFloor, resolvedActiveFloor])

  const selectedPdfCount = files.filter((file) => file.name.toLowerCase().endsWith(".pdf")).length
  const selectedImageCount = files.length - selectedPdfCount
  const uploadValidationMessage =
    files.length === 0
      ? ""
      : selectedPdfCount > 0
        ? `PDF page count should match ${selectedFloorCount} selected floor(s). Maximum supported floor count is 3.`
        : selectedImageCount !== selectedFloorCount
          ? `${selectedFloorCount} floor(s) selected, but ${selectedImageCount} image file(s) added.`
          : ""
  const uploadValidationTone = uploadValidationMessage
    ? selectedPdfCount > 0
      ? "info"
      : "error"
    : "success"

  const localPlanImage = useMemo(() => {
    if (!activeFloorData) return null
    if (activeFloorData.image_url?.startsWith("/demo/")) return activeFloorData.image_url
    return buildLocalPlanSvgDataUrl(activeFloorData)
  }, [activeFloorData])

  const landingPlanPreviewImage = useMemo(() => {
    return localDemoProject.floors?.[0]?.image_url || null
  }, [localDemoProject])

  const currentSummary = activeFloor === "building"
    ? project?.summary || {}
    : activeFloorData?.summary || {}
  const projectHeight = Number(project?.building_height || floors.length * floorHeight || 0).toFixed(1)
  const isMediumViewport = viewportWidth < 1360
  const isNarrowViewport = viewportWidth < 1080
  const isLandingCompact = viewportWidth < 1180
  const debugPanelData = {
    apiBaseUrl: API,
    projectId: projectId || project?.project_id || null,
    floorsLength: floors.length,
    activeFloor,
    floor0PolygonLength: floors?.[0]?.polygon?.length || 0,
    floor0InnerWallsLength: floors?.[0]?.inner_walls?.length || 0,
    floor0DoorsLength: floors?.[0]?.doors?.length || 0,
    floor0WindowsLength: floors?.[0]?.windows?.length || 0,
    floor0HasImageUrl: Boolean(floors?.[0]?.image_url),
    hasGeometryMeta: Boolean(geometryMeta),
  }

  const analysisSteps = [
    { label: "Polygon detected", value: currentSummary.polygon_points ?? project?.floor_count ?? 0 },
    { label: "Rooms estimated", value: currentSummary.room_count ?? 0 },
    { label: "Doors estimated", value: currentSummary.door_count ?? 0 },
    { label: "Windows estimated", value: currentSummary.window_count ?? 0 },
  ]

  const downloadJson = () => {
    if (!project) return
    const blob = new Blob([JSON.stringify(project, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${project.project_id || "plan2mass"}-analysis.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.setTimeout(() => URL.revokeObjectURL(url), 1200)
    pushToast("Analysis JSON downloaded", "success")
  }

  const downloadScreenshot = () => {
    try {
      if (!viewerRef.current) return
      const dataUrl = viewerRef.current.screenshot()
      const link = document.createElement("a")
      link.href = dataUrl
      link.download = `${project?.project_id || "plan2mass"}-screenshot.png`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      pushToast("Screenshot saved", "success")
    } catch (error) {
      pushToast("Screenshot could not be created", "error")
    }
  }

  const openViewer = async () => {
    setActiveFloor("building")
    const panel = viewerPanelRef.current
    if (!panel) return
    if (!document.fullscreenElement) {
      await panel.requestFullscreen()
    }
    pushToast("Browser viewer opened", "success")
  }

  const toggleFullscreen = async () => {
    const panel = viewerPanelRef.current
    if (!panel) return
    if (document.fullscreenElement) {
      await document.exitFullscreen()
      return
    }
    await panel.requestFullscreen()
  }

  useEffect(() => {
    if (!canvasRef.current || !project || !geometryMeta) return

    let viewer = null

    try {
      console.log("[App->createBabylonViewer] input", {
        activeFloor,
        resolvedActiveFloor,
        hasProject: Boolean(project),
        hasGeometryMeta: Boolean(geometryMeta),
        floorsLength: floors.length,
        currentFloorPolygonLength: activeFloorData?.polygon?.length || 0,
        currentFloorInnerWallsLength: activeFloorData?.inner_walls?.length || 0,
        currentFloorDoorsLength: activeFloorData?.doors?.length || 0,
        currentFloorWindowsLength: activeFloorData?.windows?.length || 0,
      })

      viewer = createBabylonViewer({
        canvas: canvasRef.current,
        project: viewerProject,
        geometryMeta,
        activeFloor: resolvedActiveFloor,
        floorHeight,
        visualOverrides: colorPaletteApplied,
        debugOptions: {
          enabled: showGeometryDebug,
          logTopology: true,
        },
      })

      viewerRef.current = viewer
      sceneRef.current = viewer.scene
      cameraRef.current = viewer.camera
      controlsRef.current = null
      modelRef.current = viewer.root
      setError("")
    } catch (sceneError) {
      console.error("Babylon scene init error:", sceneError)
      setError(sceneError?.message || "3D scene could not be created")
    }

    return () => {
      disposeBabylonViewer(viewer)
      viewerRef.current = null
      sceneRef.current = null
      cameraRef.current = null
      controlsRef.current = null
      modelRef.current = null
    }
  }, [project, viewerProject, geometryMeta, activeFloor, resolvedActiveFloor, floorHeight, showGeometryDebug, colorPaletteApplied])

  if (routePath !== "/app") {
    return (
      <div style={landingShellStyle}>
        <div style={backgroundGlowOne} />
        <div style={backgroundGlowTwo} />
        <div style={landingPageStyle}>
          <div style={{ ...landingTopBarStyle, ...(isLandingCompact ? landingTopBarCompactStyle : {}) }}>
            <div>
              <div style={brandStyle}>Plan2Mass</div>
              <div style={brandSubStyle}>2D Architectural Plan to Intelligent 3D Building</div>
            </div>
            <button style={getProtectedButtonStyle(buttonStyleSecondary, canEnterApp)} onClick={() => navigateTo("/app")} disabled={!canEnterApp}>
              ✦ Open App
            </button>
          </div>

          <div style={{ ...landingHeroGridStyle, ...(isLandingCompact ? landingHeroGridCompactStyle : {}) }}>
            <div>
              <div style={landingBadgeStyle}>Architectural Intelligence Platform</div>
              <h1 style={landingTitleStyle}>Turn 2D Plans Into Real 3D Buildings</h1>
              <p style={landingTextStyle}>
                Upload your architectural plan and instantly generate a realistic 3D model
                with room separation, wall extraction, and immersive building previews.
              </p>
              <div style={landingActionRowStyle}>
                <button style={getProtectedButtonStyle(buttonStylePrimary, canEnterApp)} onClick={() => navigateTo("/app")} disabled={!canEnterApp}>
                  ✦ Try Demo
                </button>
                <button style={getProtectedButtonStyle(buttonStyleSecondary, canEnterApp)} onClick={() => navigateTo("/app")} disabled={!canEnterApp}>
                  ⤴ Upload Plan
                </button>
              </div>

              <div style={{ ...landingSplitCardStyle, marginTop: 16, padding: 16, borderRadius: 18 }}>
                <div style={landingInfoTitleStyle}>{authMode === "register" ? "Create account" : "Sign in"}</div>
                <div style={{ display: "grid", gap: 8 }}>
                  {authMode === "register" ? (
                    <input
                      value={authName}
                      onChange={(e) => {
                        setAuthName(e.target.value)
                        if (authError) setAuthError("")
                      }}
                      placeholder="Display name"
                      style={authInputStyle}
                    />
                  ) : null}
                  <input
                    value={authEmail}
                    onChange={(e) => {
                      setAuthEmail(e.target.value)
                      if (authError) setAuthError("")
                    }}
                    placeholder="Email"
                    type="email"
                    style={authInputStyle}
                  />
                  <input
                    value={authPassword}
                    onChange={(e) => {
                      setAuthPassword(e.target.value)
                      if (authError) setAuthError("")
                    }}
                    placeholder="Password"
                    type="password"
                    style={authInputStyle}
                  />
                  {authMode === "register" ? (
                    <input
                      value={authPasswordConfirm}
                      onChange={(e) => {
                        setAuthPasswordConfirm(e.target.value)
                        if (authError) setAuthError("")
                      }}
                      placeholder="Confirm password"
                      type="password"
                      style={authInputStyle}
                    />
                  ) : null}
                  <div style={{ ...helperTextStyle, marginTop: 0 }}>
                    {authMode === "register"
                      ? "Create an account to save your projects and continue from any session."
                      : "Sign in to keep project history and secure model access."}
                  </div>
                  {!authValidation.isValid ? (
                    <div style={{ ...helperTextStyle, color: "#ffd9a8", marginTop: 0 }}>
                      {authValidation.issues[0]}
                    </div>
                  ) : null}
                  {authError ? <div style={{ ...helperTextStyle, color: "#ffb4b4" }}>{authError}</div> : null}
                  {authSuccess ? <div style={{ ...helperTextStyle, color: "#b7ffd2" }}>{authSuccess}</div> : null}
                  <div style={{ display: "flex", gap: 8 }}>
                    <button
                      style={buttonStylePrimary}
                      onClick={submitAuth}
                      disabled={isAuthLoading || !authValidation.isValid}
                    >
                      {isAuthLoading ? "Please wait..." : authMode === "register" ? "Register" : "Login"}
                    </button>
                    <button
                      style={buttonStyleSecondary}
                      onClick={() => {
                        setAuthError("")
                        setAuthSuccess("")
                        setAuthMode((m) => (m === "register" ? "login" : "register"))
                      }}
                    >
                      {authMode === "register" ? "Have account" : "Create account"}
                    </button>
                  </div>
                </div>
              </div>

              <div style={featureStripStyle}>
                {[
                  "Room Detection",
                  "Window & Door Placement",
                  "Real-time 3D Visualization",
                ].map((item) => (
                  <div key={item} style={featureStripCardStyle}>
                    {item}
                  </div>
                ))}
              </div>
            </div>

          </div>

          <div style={landingSectionStyle}>
            <div style={landingSectionHeaderStyle}>How It Works</div>
            <div style={{ ...landingCardGridStyle, ...(isLandingCompact ? landingCardGridCompactStyle : {}) }}>
              {[
                ["01", "Upload Plan", "Drop your floor plans and let the pipeline prepare them for analysis."],
                ["02", "Detect Structure", "Extract walls, openings, and room regions from the 2D drawing."],
                ["03", "Generate 3D Model", "Preview a navigable architectural model directly in the browser."],
              ].map(([step, title, text]) => (
                <div key={step} style={landingInfoCardStyle}>
                  <div style={landingStepPillStyle}>{step}</div>
                  <div style={landingInfoTitleStyle}>{title}</div>
                  <div style={landingInfoTextStyle}>{text}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ ...landingSplitSectionStyle, ...(isLandingCompact ? landingSplitSectionCompactStyle : {}) }}>
            <div style={landingSplitCardStyle}>
              <div style={panelEyebrowStyle}>Before</div>
              <div style={landingInfoTitleStyle}>2D Architectural Plan</div>
              <img src={landingPlanPreviewImage || "/landing/plan-2d-sample.svg"} alt="2D plan" style={landingBeforeAfterImageStyle} />
            </div>
            <div style={landingSplitCardStyle}>
              <div style={panelEyebrowStyle}>After</div>
              <div style={landingInfoTitleStyle}>Interactive 3D Building</div>
              <img src="/landing/demo-local3d.png" alt="3D building" style={landingBeforeAfterImageStyle} />
            </div>
          </div>

          <div style={landingCtaStyle}>
            <div>
              <div style={landingSectionHeaderStyle}>Start Building Now</div>
              <div style={landingInfoTextStyle}>
                Explore the live application and turn floor plans into presentation-ready 3D building views.
              </div>
            </div>
            <button style={getProtectedButtonStyle(buttonStylePrimary, canEnterApp)} onClick={() => navigateTo("/app")} disabled={!canEnterApp}>
              → Go To App
            </button>
          </div>
        </div>
      </div>
    )
  }

  if (!project) {
    return (
      <div style={shellStyle}>
        <div style={heroStyle}>
          <div style={badgeStyle}>Plan2Mass</div>
          <h1 style={heroTitleStyle}>Clean 3D building models from 2D architectural plans</h1>
          <p style={heroTextStyle}>
            Strong plan analysis, cleaner opening placement, and a more architectural presentation layer.
          </p>
          <div style={loadingCardStyle}>
            {error ? error : "Loading project..."}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div style={shellStyle}>
      <div style={backgroundGlowOne} />
      <div style={backgroundGlowTwo} />

      <div style={pageStyle}>
        <div style={toastStackStyle}>
          {toasts.map((toast) => (
            <div
              key={toast.id}
              style={{
                ...toastStyle,
                ...(toast.tone === "success" ? toastSuccessStyle : {}),
                ...(toast.tone === "error" ? toastErrorStyle : {}),
              }}
            >
              {toast.message}
            </div>
          ))}
        </div>

        <div style={devDebugPanelStyle}>
          <div style={devDebugPanelTitleStyle}>Debug</div>
          <div style={devDebugPanelTextStyle}>API: {debugPanelData.apiBaseUrl}</div>
          <div style={devDebugPanelTextStyle}>project_id: {String(debugPanelData.projectId)}</div>
          <div style={devDebugPanelTextStyle}>floors: {debugPanelData.floorsLength}</div>
          <div style={devDebugPanelTextStyle}>activeFloor: {String(debugPanelData.activeFloor)}</div>
          <div style={devDebugPanelTextStyle}>floor0 polygon: {debugPanelData.floor0PolygonLength}</div>
          <div style={devDebugPanelTextStyle}>floor0 inner_walls: {debugPanelData.floor0InnerWallsLength}</div>
          <div style={devDebugPanelTextStyle}>floor0 doors: {debugPanelData.floor0DoorsLength}</div>
          <div style={devDebugPanelTextStyle}>floor0 windows: {debugPanelData.floor0WindowsLength}</div>
          <div style={devDebugPanelTextStyle}>floor0 image_url: {debugPanelData.floor0HasImageUrl ? "yes" : "no"}</div>
          <div style={devDebugPanelTextStyle}>geometryMeta: {debugPanelData.hasGeometryMeta ? "yes" : "no"}</div>
        </div>

        <div style={{ ...topBarStyle, ...(isNarrowViewport ? topBarCompactStyle : {}) }}>
          <div>
            <div style={brandStyle}>Plan2Mass</div>
            <div style={brandSubStyle}>2D Architectural Plan to Intelligent 3D Building</div>
          </div>

          <div style={{ ...topBarRightStyle, ...(isNarrowViewport ? topBarRightCompactStyle : {}) }}>
            <div style={headerActionPillStyle}>
              <button style={headerIconButtonStyle} onClick={openViewer}>◫ Viewer</button>
              <button style={headerIconButtonStyle} onClick={downloadScreenshot}>⌁ Screenshot</button>
              <button style={headerIconButtonStyle} onClick={downloadModel}>⬒ GLB</button>
            </div>

            <div style={profileCardStyle}>
              <div style={profileAvatarStyle}>
                {(authUser?.display_name || authUser?.email || "GU")
                  .split(" ")
                  .map((part) => part[0])
                  .join("")
                  .slice(0, 2)
                  .toUpperCase()}
              </div>
              <div>
                <div style={profileLabelStyle}>User</div>
                <div style={profileNameStyle}>{authUser?.display_name || authUser?.email || "Guest User"}</div>
              </div>
              {authUser ? (
                <button style={{ ...buttonStyleSecondary, padding: "8px 10px" }} onClick={logout}>
                  Logout
                </button>
              ) : null}
            </div>
          </div>
        </div>

        <div
          style={{
            ...mainGridStyle,
            ...(isMediumViewport ? mainGridMediumStyle : {}),
            ...(isNarrowViewport ? mainGridNarrowStyle : {}),
          }}
        >
          <div style={sidebarCardStyle}>
            <div style={sectionTitleStyle}>Control Center</div>

            <div style={controlSectionStyle}>
              <button style={collapsibleHeaderStyle} onClick={() => toggleSidebarSection("upload")}>
                <span>⤴ Upload</span>
                <span>{sidebarSections.upload ? "−" : "+"}</span>
              </button>
              {sidebarSections.upload ? (
                <div style={controlBlockStyle}>
                  <div style={inputLabelStyle}>Upload floor plans</div>
                  <div style={controlHeaderRowStyle}>
                    <div style={inputLabelStyle}>Expected floors</div>
                    <div style={metricPillStyle}>Max 3 floors</div>
                  </div>
                  <div style={floorCountGridStyle}>
                    {[1, 2, 3].map((count) => (
                      <button
                        key={count}
                        type="button"
                        onClick={() => setSelectedFloorCount(count)}
                        style={{
                          ...floorCountButtonStyle,
                          ...(selectedFloorCount === count
                            ? activeFloorCountButtonStyle
                            : inactiveFloorCountButtonStyle),
                        }}
                      >
                        {count} Floor{count > 1 ? "s" : ""}
                      </button>
                    ))}
                  </div>
                  <label
                    style={{ ...dropzoneStyle, ...(isDropActive ? dropzoneActiveStyle : {}) }}
                    onDragEnter={(event) => {
                      event.preventDefault()
                      setIsDropActive(true)
                    }}
                    onDragOver={(event) => {
                      event.preventDefault()
                      setIsDropActive(true)
                    }}
                    onDragLeave={(event) => {
                      event.preventDefault()
                      setIsDropActive(false)
                    }}
                    onDrop={(event) => {
                      event.preventDefault()
                      setIsDropActive(false)
                      const dropped = [...(event.dataTransfer?.files || [])]
                      if (dropped.length) {
                        setFiles(dropped)
                        pushToast(`${dropped.length} file(s) selected`, "success")
                      }
                    }}
                  >
                    <input
                      type="file"
                      multiple
                      onChange={(e) => {
                        const selected = [...e.target.files]
                        setFiles(selected)
                        if (selected.length) pushToast(`${selected.length} file(s) selected`, "success")
                      }}
                      style={hiddenFileInputStyle}
                    />
                    <div style={dropzoneTitleStyle}>Drop floor plan files here</div>
                    <div style={dropzoneTextStyle}>PNG, JPG, and PDF supported. Click to browse.</div>
                    {files.length ? (
                      <div style={fileListStyle}>
                        {files.slice(0, 3).map((file) => (
                          <div key={`${file.name}-${file.size}`} style={fileChipStyle}>
                            {file.name}
                          </div>
                        ))}
                      </div>
                    ) : null}
                  </label>

                  <div style={actionRowStyle}>
                    <button style={buttonStylePrimary} onClick={uploadPlan} disabled={isLoading}>
                      Upload Plan
                    </button>
                    <button
                      style={buttonStyleSecondary}
                      onClick={refreshAnalysis}
                      disabled={isLoading || !projectId}
                    >
                      Refresh
                    </button>
                    <button style={buttonStyleGhost} onClick={resetToDemoProject} disabled={isLoading}>
                      Reset to Demo
                    </button>
                  </div>
                  <div style={helperTextStyle}>
                    Open Viewer shows the model in-browser. Download GLB is for Blender, Windows 3D Viewer, or online viewers.
                  </div>
                  <div style={helperTextStyle}>
                    If you select 2 floors, upload both floor images together or use a 2-page PDF. The current version does not add floors one by one after upload.
                  </div>
                  {uploadValidationMessage ? (
                    <div
                      style={{
                        ...validationNoteStyle,
                        ...(uploadValidationTone === "error"
                          ? validationErrorStyle
                          : uploadValidationTone === "success"
                            ? validationSuccessStyle
                            : validationInfoStyle),
                      }}
                    >
                      {uploadValidationMessage}
                    </div>
                  ) : null}
                  <div style={guidancePanelStyle}>
                    <div style={guidanceTitleStyle}>Plan preparation rules</div>
                    <div style={guidanceListStyle}>
                      <div style={guidanceItemStyle}>Walls should be clearly visible and continuous.</div>
                      <div style={guidanceItemStyle}>Avoid furniture, dimensions, grid lines, and annotations.</div>
                      <div style={guidanceItemStyle}>Upload one plan image per floor, or a PDF whose pages match the selected floor count.</div>
                      <div style={guidanceItemStyle}>The current version supports 1 to 3 floors.</div>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>

            <div style={controlSectionStyle}>
              <button style={collapsibleHeaderStyle} onClick={() => toggleSidebarSection("settings")}>
                <span>⌘ Settings</span>
                <span>{sidebarSections.settings ? "−" : "+"}</span>
              </button>
              {sidebarSections.settings ? (
                <div style={controlBlockStyle}>
                  <div style={controlHeaderRowStyle}>
                    <div style={inputLabelStyle}>Floor height</div>
                    <div style={metricPillStyle}>{floorHeight.toFixed(1)} m</div>
                  </div>
                  <input
                    type="range"
                    min="2.4"
                    max="5.0"
                    step="0.1"
                    value={floorHeight}
                    onChange={(e) => setFloorHeight(Number(e.target.value))}
                    style={rangeStyle}
                  />
                </div>
              ) : null}
            </div>

            <div style={controlSectionStyle}>
              <button style={collapsibleHeaderStyle} onClick={() => toggleSidebarSection("floors")}>
                <span>▣ Floors</span>
                <span>{sidebarSections.floors ? "−" : "+"}</span>
              </button>
              {sidebarSections.floors ? (
                <div style={controlBlockStyle}>
                  <div style={inputLabelStyle}>View mode</div>
                  <div style={tabGridStyle}>
                    {floors.map((_, i) => (
                      <button
                        key={i}
                        onClick={() => setActiveFloor(i)}
                        style={{
                          ...tabButtonStyle,
                          ...(activeFloor === i ? activeTabButtonStyle : inactiveTabButtonStyle),
                        }}
                      >
                        Floor {i + 1}
                      </button>
                    ))}

                    <button
                      onClick={() => setActiveFloor("building")}
                      style={{
                        ...tabButtonStyle,
                        ...(activeFloor === "building" ? activeTabButtonStyle : inactiveTabButtonStyle),
                      }}
                    >
                      Building
                    </button>
                  </div>
                  {activeFloor === "building" ? (
                    <div style={{ marginTop: 12 }}>
                      <div style={inputLabelStyle}>Building mode</div>
                      <div style={tabGridStyle}>
                        <button
                          style={{
                            ...tabButtonStyle,
                            ...activeTabButtonStyle,
                          }}
                        >
                          Normal
                        </button>
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>
            <div style={controlSectionStyle}>
              <div style={inputLabelStyle}>Color palette</div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 10 }}>
                <button style={{ ...tabButtonStyle, ...inactiveTabButtonStyle }} onClick={() => applyPalettePreset("warm")}>Warm</button>
                <button style={{ ...tabButtonStyle, ...inactiveTabButtonStyle }} onClick={() => applyPalettePreset("modern")}>Modern</button>
                <button style={{ ...tabButtonStyle, ...inactiveTabButtonStyle }} onClick={() => applyPalettePreset("light")}>Light</button>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                {[
                  ["Walls", "walls"],
                  ["Slabs", "slabs"],
                  ["Roof", "roof"],
                  ["Frames", "frames"],
                  ["Glass", "glass"],
                  ["Door", "door"],
                  ["Ground", "ground"],
                ].map(([label, key]) => (
                  <label key={key} style={{ display: "grid", gap: 4, color: "#d7e4eb", fontSize: 12 }}>
                    <span>{label}</span>
                    <input
                      type="color"
                      value={colorPaletteDraft[key]}
                      onChange={(event) =>
                        setColorPaletteDraft((current) => ({
                          ...current,
                          [key]: event.target.value,
                        }))
                      }
                      style={{
                        width: "100%",
                        height: 34,
                        borderRadius: 8,
                        border: "1px solid rgba(255,255,255,0.1)",
                        background: "transparent",
                        cursor: "pointer",
                      }}
                    />
                  </label>
                ))}
              </div>
              <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
                <button
                  style={{ ...tabButtonStyle, ...activeTabButtonStyle }}
                  onClick={() => setColorPaletteApplied(colorPaletteDraft)}
                >
                  Apply
                </button>
                <button
                  style={{ ...tabButtonStyle, ...inactiveTabButtonStyle }}
                  onClick={() => setColorPaletteDraft(colorPaletteApplied)}
                >
                  Revert
                </button>
                <button
                  style={{ ...tabButtonStyle, ...inactiveTabButtonStyle }}
                  onClick={() => {
                    setColorPaletteDraft(defaultPalette)
                    setColorPaletteApplied(defaultPalette)
                  }}
                >
                  Reset
                </button>
              </div>
            </div>
          </div>

          <div
            style={{
              ...workspaceColumnStyle,
              ...(isMediumViewport ? { gridColumn: "2" } : {}),
              ...(isNarrowViewport ? { gridColumn: "auto" } : {}),
            }}
          >
            <div style={planCardStyle}>
              <div style={panelHeaderStyle}>
                <div>
                  <div style={panelEyebrowStyle}>Reference</div>
                  <div style={panelTitleStyle}>2D Floor Plan</div>
                </div>
              </div>
              <div style={panelBodyStyle}>
                <div style={planStageStyle}>
                  {currentPlanImage ? (
                    <img
                      src={
                        currentPlanImage.startsWith("data:") || currentPlanImage.startsWith("/demo/")
                          ? currentPlanImage
                          : `${API}${currentPlanImage}`
                      }
                      alt="2D plan"
                      style={planImageStyle}
                    />
                  ) : localPlanImage ? (
                    <img src={localPlanImage} alt="2D plan" style={planImageStyle} />
                  ) : (
                    <div style={emptyPanelStyle}>Plan image not available</div>
                  )}

                  {showGeometryDebug && activePlanDebug ? (
                    <svg viewBox={activePlanDebug.viewBox} style={planOverlayStyle}>
                      {activePlanDebug.graph.walls.map((wall) => {
                        const a = activePlanDebug.mapPoint(wall.line[0], wall.line[1])
                        const b = activePlanDebug.mapPoint(wall.line[2], wall.line[3])
                        return (
                          <line
                            key={`dbg-wall-${wall.id}`}
                            x1={a.x}
                            y1={a.y}
                            x2={b.x}
                            y2={b.y}
                            stroke={wall.kind === "outer" ? "#8bd5ff" : "#ffd479"}
                            strokeWidth={wall.kind === "outer" ? 5 : 3}
                            strokeLinecap="round"
                            opacity={0.95}
                          />
                        )
                      })}

                      {activePlanDebug.matched.map((opening) => {
                        const p = activePlanDebug.mapPoint(opening.point.x, opening.point.y)
                        return (
                          <circle
                            key={`dbg-opening-${opening.id}`}
                            cx={p.x}
                            cy={p.y}
                            r={opening.type === "door" ? 7 : 6}
                            fill={opening.type === "door" ? "#ff7d7d" : "#8cffbf"}
                            stroke="#0f1720"
                            strokeWidth={2}
                          />
                        )
                      })}
                    </svg>
                  ) : null}
                </div>
              </div>
            </div>

            <div style={viewerCardStyle} ref={viewerPanelRef}>
              <div style={panelHeaderStyle}>
                <div>
                  <div style={panelEyebrowStyle}>Output</div>
                  <div style={panelTitleStyle}>3D Building View</div>
                </div>
                <div style={viewerActionRowStyle}>
                  <button style={viewerActionButtonStyle} onClick={openViewer}>◫ Viewer</button>
                  <button style={viewerActionButtonStyle} onClick={downloadScreenshot}>⌁ Screenshot</button>
                  <button style={viewerActionButtonStyle} onClick={downloadModel}>⬒ GLB</button>
                </div>
              </div>
              <div style={canvasWrapStyle}>
                <canvas ref={canvasRef} style={canvasStyle} />
                <div style={canvasHintStyle}>Rotate: mouse drag · Zoom: scroll</div>
              </div>
            </div>
          </div>

          <div
            style={{
              ...rightRailStyle,
              ...(isMediumViewport ? { gridColumn: "1 / -1", gridTemplateColumns: "repeat(2, minmax(0, 1fr))" } : {}),
              ...(isNarrowViewport ? { gridColumn: "auto", gridTemplateColumns: "1fr" } : {}),
            }}
          >
            <div style={infoPanelStyle}>
              <div style={infoTitleStyle}>Action Center</div>
              <div style={actionRailStyle}>
                <button style={buttonStyleSecondary} onClick={openViewer}>◫ Open Viewer</button>
                <button style={buttonStyleSecondary} onClick={downloadScreenshot}>⌁ Save Screenshot</button>
                <button style={buttonStyleGhost} onClick={downloadModel}>⬒ Download GLB</button>
                <button style={buttonStyleSecondary} onClick={downloadJson}>⎘ Download JSON</button>
                <button
                  style={showGeometryDebug ? buttonStyleGhost : buttonStyleSecondary}
                  onClick={() => setShowGeometryDebug((value) => !value)}
                >
                  {showGeometryDebug ? "Debug: ON" : "Debug: OFF"}
                </button>
              </div>
            </div>

            <div style={infoPanelStyle}>
              <div style={infoTitleStyle}>Project Snapshot</div>
              <div style={infoTextStyle}>Project: {projectId || project.project_id || "N/A"}</div>
              <div style={infoTextStyle}>Floors: {project.floor_count || floors.length}</div>
              <div style={infoTextStyle}>Height: {projectHeight} m</div>
              <div style={infoTextStyle}>Rooms: {project?.summary?.room_count ?? 0}</div>
            </div>

            <div style={infoPanelStyle}>
              <div style={infoTitleStyle}>My Model History</div>
              {!authUser ? (
                <div style={infoTextStyle}>Sign in to keep and list your project history.</div>
              ) : userProjects.length === 0 ? (
                <div style={infoTextStyle}>No saved models yet. Upload a plan to start.</div>
              ) : (
                <div style={{ display: "grid", gap: 8 }}>
                  {userProjects.slice(0, 8).map((item) => (
                    <button
                      key={`${item.project_id}-${item.created_at}`}
                      style={{ ...buttonStyleSecondary, textAlign: "left" }}
                      onClick={() => loadProject(item.project_id)}
                    >
                      <div style={{ fontWeight: 700 }}>{item.project_id}</div>
                      <div style={{ fontSize: 12, opacity: 0.8 }}>
                        Floors: {item.floor_count} • {new Date(item.created_at).toLocaleString()}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div style={statsGridStyle}>
              <div style={statCardStyle}>
                <div style={statLabelStyle}>Rooms</div>
                <div style={statValueStyle}>{currentSummary.room_count ?? 0}</div>
              </div>
              <div style={statCardStyle}>
                <div style={statLabelStyle}>Inner Walls</div>
                <div style={statValueStyle}>{currentSummary.inner_wall_count ?? 0}</div>
              </div>
              <div style={statCardStyle}>
                <div style={statLabelStyle}>Doors</div>
                <div style={statValueStyle}>{currentSummary.door_count ?? 0}</div>
              </div>
              <div style={statCardStyle}>
                <div style={statLabelStyle}>Windows</div>
                <div style={statValueStyle}>{currentSummary.window_count ?? 0}</div>
              </div>
            </div>

            <div style={progressPanelStyle}>
              <div style={infoTitleStyle}>Analysis Steps</div>
              {analysisSteps.map((step) => (
                <div key={step.label} style={progressRowStyle}>
                  <span>{step.label}</span>
                  <span style={progressBadgeStyle}>{step.value}</span>
                </div>
              ))}
            </div>

            <div style={infoPanelStyle}>
              <div style={infoTitleStyle}>Pipeline</div>
              <div style={infoTextStyle}>
                Outer polygon, inner walls, doors, and windows come from backend JSON outputs.
              </div>
              <div style={infoTextStyle}>
                Wall matching and cleaner segmentation are applied before Babylon building generation.
              </div>
            </div>

            {error ? <div style={errorCardStyle}>{error}</div> : null}
            {isLoading ? (
              <div style={loadingInlineStyle}>
                <div style={loadingDotStyle} />
                {loadingPhases[loadingPhaseIndex]}
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  )
}

function setupArchitecturalScene(scene, { activeFloor } = {}) {
  if (!sceneSkyTextureCache) {
    sceneSkyTextureCache = createSkyTexture()
    sceneSkyTextureCache.mapping = THREE.EquirectangularReflectionMapping
  }
  if (!sceneGroundTextureCache) {
    const concreteSet = getConcreteTextureSet()
    sceneGroundTextureCache = concreteSet.map
    sceneGroundNormalTextureCache = concreteSet.normalMap
    sceneGroundRoughnessTextureCache = concreteSet.roughnessMap
  }

  const isBuildingView = activeFloor === "building"
  scene.environment = sceneHdriTextureCache || sceneSkyTextureCache

  if (isBuildingView && sceneHdriTextureCache) {
    scene.background = sceneHdriTextureCache
  }

  if (isBuildingView && !sceneHdriTextureCache && !sceneHdriLoadStarted) {
    sceneHdriLoadStarted = true
    new RGBELoader().load(
      "/hdr/sky.hdr",
      (texture) => {
        texture.mapping = THREE.EquirectangularReflectionMapping
        sceneHdriTextureCache = texture
        if (sceneRef.current) {
          sceneRef.current.environment = texture
          sceneRef.current.background = texture
        }
      },
      undefined,
      () => {
        sceneHdriLoadStarted = false
      }
    )
  }

  const ambient = new THREE.AmbientLight(0xffffff, isBuildingView ? 0.16 : 0.52)
  scene.add(ambient)

  const hemi = new THREE.HemisphereLight(0xf6f8fb, 0xb39a82, isBuildingView ? 1.08 : 1.28)
  hemi.position.set(0, 60, 0)
  scene.add(hemi)

  const sun = new THREE.DirectionalLight(0xffefcf, isBuildingView ? 3.3 : 2.35)
  sun.position.set(28, 31, 18)
  sun.castShadow = true
  sun.shadow.mapSize.width = 4096
  sun.shadow.mapSize.height = 4096
  sun.shadow.camera.near = 1
  sun.shadow.camera.far = 180
  sun.shadow.camera.left = -56
  sun.shadow.camera.right = 56
  sun.shadow.camera.top = 56
  sun.shadow.camera.bottom = -56
  sun.shadow.radius = 2.2
  sun.shadow.blurSamples = 8
  sun.shadow.bias = -0.00014
  scene.add(sun)

  const fill = new THREE.DirectionalLight(0xdbe7f1, isBuildingView ? 0.68 : 0.96)
  fill.position.set(-18, 18, -22)
  scene.add(fill)

  const rim = new THREE.DirectionalLight(0xf4d8bb, isBuildingView ? 0.34 : 0.48)
  rim.position.set(10, 14, 26)
  scene.add(rim)

  const skyDome = new THREE.Mesh(
    new THREE.SphereGeometry(180, 32, 16),
    new THREE.MeshBasicMaterial({
      map: sceneSkyTextureCache,
      side: THREE.BackSide,
      depthWrite: false,
    })
  )
  skyDome.position.y = 18
  scene.add(skyDome)

  const pad = new THREE.Mesh(
    new THREE.BoxGeometry(28, 0.18, 22),
    new THREE.MeshStandardMaterial({
      color: "#d7d0c5",
      roughness: 0.9,
      metalness: 0.0,
    })
  )
  pad.position.set(0, -0.1, 0.4)
  pad.receiveShadow = true
  scene.add(pad)

  const padBorder = new THREE.Mesh(
    new THREE.BoxGeometry(29.2, 0.06, 23.2),
    new THREE.MeshStandardMaterial({
      color: "#c6bcae",
      roughness: 0.92,
      metalness: 0.0,
    })
  )
  padBorder.position.set(0, -0.18, 0.4)
  padBorder.receiveShadow = true
  scene.add(padBorder)

  const siteGeo = new THREE.PlaneGeometry(260, 260, 1, 1)
  const siteMat = new THREE.MeshStandardMaterial({
    map: sceneGroundTextureCache,
    normalMap: sceneGroundNormalTextureCache,
    roughnessMap: sceneGroundRoughnessTextureCache,
    color: "#ece5db",
    roughness: 0.98,
    metalness: 0,
  })
  const site = new THREE.Mesh(siteGeo, siteMat)
  site.rotation.x = -Math.PI / 2
  site.position.y = -0.58
  site.receiveShadow = true
  scene.add(site)

  const contactShadow = new THREE.Mesh(
    new THREE.CircleGeometry(12.5, 64),
    new THREE.MeshBasicMaterial({
      color: "#7b746a",
      transparent: true,
      opacity: isBuildingView ? 0.18 : 0.06,
      depthWrite: false,
    })
  )
  contactShadow.rotation.x = -Math.PI / 2
  contactShadow.position.set(0, -0.53, 0.25)
  scene.add(contactShadow)

}

function disposeScene(scene) {
  scene.traverse((obj) => {
    if (obj.geometry) {
      obj.geometry.dispose()
    }

    if (obj.material) {
      if (Array.isArray(obj.material)) {
        obj.material.forEach((m) => m.dispose())
      } else {
        obj.material.dispose()
      }
    }
  })
}

function buildProjectGeometryMeta(project, targetSize = 22) {
  if (!project?.floors?.length) {
    console.warn("GEOMETRY_META_NULL - 3D viewer cannot build")
    console.log("[buildProjectGeometryMeta] summary", {
      geometryMetaNull: true,
      allPointsCount: 0,
      floorCount: 0,
      polygonPointCountPerFloor: [],
      bbox: null,
    })
    return null
  }

  const allPoints = []
  const polygonPointCountPerFloor = []

  project.floors.forEach((floor) => {
    polygonPointCountPerFloor.push(floor?.polygon?.length || 0)
    ;(floor.polygon || []).forEach((p) => {
      if (Array.isArray(p) && p.length >= 2) {
        allPoints.push({ x: p[0], y: p[1] })
      }
    })
  })

  if (!allPoints.length) {
    console.warn("GEOMETRY_META_NULL - 3D viewer cannot build")
    console.log("[buildProjectGeometryMeta] summary", {
      geometryMetaNull: true,
      allPointsCount: 0,
      floorCount: project.floors.length,
      polygonPointCountPerFloor,
      bbox: null,
    })
    return null
  }

  const minX = Math.min(...allPoints.map((p) => p.x))
  const maxX = Math.max(...allPoints.map((p) => p.x))
  const minY = Math.min(...allPoints.map((p) => p.y))
  const maxY = Math.max(...allPoints.map((p) => p.y))
  const transform = buildWorldTransform({
    minX,
    maxX,
    minY,
    maxY,
    targetSize,
  })

  console.log("[buildProjectGeometryMeta] summary", {
    geometryMetaNull: false,
    allPointsCount: allPoints.length,
    floorCount: project.floors.length,
    polygonPointCountPerFloor,
    bbox: {
      minX,
      maxX,
      minY,
      maxY,
    },
  })

  return {
    minX,
    maxX,
    minY,
    maxY,
    ...transform,
  }
}

let interiorFloorMaterialCache = null
let sceneSkyTextureCache = null
let sceneHdriTextureCache = null
let sceneHdriLoadStarted = false
let sceneGroundTextureCache = null
let sceneGroundNormalTextureCache = null
let sceneGroundRoughnessTextureCache = null
let plasterTextureSetCache = null
let concreteTextureSetCache = null
let roofTextureSetCache = null
let darkMetalTextureSetCache = null
let woodTextureSetCache = null
let stoneTextureSetCache = null

function textureFromCanvas(canvas, options = {}) {
  const texture = new THREE.CanvasTexture(canvas)
  texture.wrapS = options.wrapS ?? THREE.RepeatWrapping
  texture.wrapT = options.wrapT ?? THREE.RepeatWrapping
  if (options.repeat) {
    texture.repeat.set(options.repeat[0], options.repeat[1])
  }
  texture.anisotropy = 8
  texture.colorSpace = options.colorSpace ?? THREE.SRGBColorSpace
  return texture
}

function buildNormalMapFromCanvas(sourceCanvas, strength = 1.8) {
  const width = sourceCanvas.width
  const height = sourceCanvas.height
  const sourceCtx = sourceCanvas.getContext("2d")
  const src = sourceCtx.getImageData(0, 0, width, height).data

  const normalCanvas = document.createElement("canvas")
  normalCanvas.width = width
  normalCanvas.height = height
  const normalCtx = normalCanvas.getContext("2d")
  const imageData = normalCtx.createImageData(width, height)

  const getValue = (x, y) => {
    const cx = Math.max(0, Math.min(width - 1, x))
    const cy = Math.max(0, Math.min(height - 1, y))
    return src[(cy * width + cx) * 4] / 255
  }

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const left = getValue(x - 1, y)
      const right = getValue(x + 1, y)
      const top = getValue(x, y - 1)
      const bottom = getValue(x, y + 1)

      const dx = (right - left) * strength
      const dy = (bottom - top) * strength
      const dz = 1
      const length = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1

      const nx = dx / length
      const ny = dy / length
      const nz = dz / length

      const idx = (y * width + x) * 4
      imageData.data[idx] = Math.round((nx * 0.5 + 0.5) * 255)
      imageData.data[idx + 1] = Math.round((ny * 0.5 + 0.5) * 255)
      imageData.data[idx + 2] = Math.round((nz * 0.5 + 0.5) * 255)
      imageData.data[idx + 3] = 255
    }
  }

  normalCtx.putImageData(imageData, 0, 0)
  return textureFromCanvas(normalCanvas, {
    repeat: [6, 6],
    colorSpace: THREE.NoColorSpace,
  })
}

function buildRoughnessMapFromCanvas(sourceCanvas, repeat = [6, 6]) {
  const width = sourceCanvas.width
  const height = sourceCanvas.height
  const sourceCtx = sourceCanvas.getContext("2d")
  const src = sourceCtx.getImageData(0, 0, width, height).data

  const roughnessCanvas = document.createElement("canvas")
  roughnessCanvas.width = width
  roughnessCanvas.height = height
  const roughnessCtx = roughnessCanvas.getContext("2d")
  const imageData = roughnessCtx.createImageData(width, height)

  for (let i = 0; i < src.length; i += 4) {
    const value = src[i]
    const adjusted = Math.max(80, Math.min(240, Math.round(value * 0.88 + 34)))
    imageData.data[i] = adjusted
    imageData.data[i + 1] = adjusted
    imageData.data[i + 2] = adjusted
    imageData.data[i + 3] = 255
  }

  roughnessCtx.putImageData(imageData, 0, 0)
  return textureFromCanvas(roughnessCanvas, {
    repeat,
    colorSpace: THREE.NoColorSpace,
  })
}

function createPlasterTextureSet() {
  const canvas = document.createElement("canvas")
  canvas.width = 1024
  canvas.height = 1024
  const ctx = canvas.getContext("2d")

  const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height)
  gradient.addColorStop(0, "#e7e0d6")
  gradient.addColorStop(0.5, "#ddd3c7")
  gradient.addColorStop(1, "#f2ece4")
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, canvas.width, canvas.height)

  for (let i = 0; i < 1300; i += 1) {
    const alpha = 0.006 + Math.random() * 0.012
    const shade = 208 + Math.floor(Math.random() * 16)
    ctx.fillStyle = `rgba(${shade},${shade - 4},${shade - 10},${alpha})`
    const size = 1 + Math.random() * 4
    ctx.beginPath()
    ctx.ellipse(
      Math.random() * canvas.width,
      Math.random() * canvas.height,
      size,
      size * (0.6 + Math.random() * 0.7),
      Math.random() * Math.PI,
      0,
      Math.PI * 2
    )
    ctx.fill()
  }

  const map = textureFromCanvas(canvas, { repeat: [5.5, 5.5] })
  const normalMap = buildNormalMapFromCanvas(canvas, 1.1)
  normalMap.repeat.set(5.5, 5.5)
  const roughnessMap = buildRoughnessMapFromCanvas(canvas, [5.5, 5.5])
  return { map, normalMap, roughnessMap }
}

function createConcreteTextureSet() {
  const canvas = document.createElement("canvas")
  canvas.width = 1024
  canvas.height = 1024
  const ctx = canvas.getContext("2d")

  ctx.fillStyle = "#cfc7bb"
  ctx.fillRect(0, 0, canvas.width, canvas.height)

  for (let i = 0; i < 3200; i += 1) {
    const alpha = 0.02 + Math.random() * 0.06
    const shade = 160 + Math.floor(Math.random() * 50)
    ctx.fillStyle = `rgba(${shade},${shade - 5},${shade - 10},${alpha})`
    const size = 2 + Math.random() * 12
    ctx.fillRect(Math.random() * canvas.width, Math.random() * canvas.height, size, size)
  }

  for (let i = 0; i < 90; i += 1) {
    ctx.strokeStyle = `rgba(140,132,122,${0.04 + Math.random() * 0.04})`
    ctx.lineWidth = 1 + Math.random() * 2.2
    ctx.beginPath()
    ctx.moveTo(Math.random() * canvas.width, Math.random() * canvas.height)
    ctx.lineTo(Math.random() * canvas.width, Math.random() * canvas.height)
    ctx.stroke()
  }

  const map = textureFromCanvas(canvas, { repeat: [10, 10] })
  const normalMap = buildNormalMapFromCanvas(canvas, 2.25)
  normalMap.repeat.set(10, 10)
  const roughnessMap = buildRoughnessMapFromCanvas(canvas, [10, 10])
  return { map, normalMap, roughnessMap }
}

function createRoofTextureSet() {
  const canvas = document.createElement("canvas")
  canvas.width = 1024
  canvas.height = 1024
  const ctx = canvas.getContext("2d")

  ctx.fillStyle = "#4d535a"
  ctx.fillRect(0, 0, canvas.width, canvas.height)

  const stripeHeight = 30
  for (let y = 0; y < canvas.height; y += stripeHeight) {
    const tone = 72 + (y / stripeHeight) % 2 * 10
    ctx.fillStyle = `rgba(${tone},${tone + 2},${tone + 6},0.55)`
    ctx.fillRect(0, y, canvas.width, stripeHeight - 2)
  }

  for (let i = 0; i < 1400; i += 1) {
    const alpha = 0.018 + Math.random() * 0.04
    const shade = 95 + Math.floor(Math.random() * 30)
    ctx.fillStyle = `rgba(${shade},${shade},${shade + 4},${alpha})`
    const size = 2 + Math.random() * 7
    ctx.fillRect(Math.random() * canvas.width, Math.random() * canvas.height, size, size)
  }

  const map = textureFromCanvas(canvas, { repeat: [4.5, 4.5] })
  const normalMap = buildNormalMapFromCanvas(canvas, 2.4)
  normalMap.repeat.set(4.5, 4.5)
  const roughnessMap = buildRoughnessMapFromCanvas(canvas, [4.5, 4.5])
  return { map, normalMap, roughnessMap }
}

function createDarkMetalTextureSet() {
  const canvas = document.createElement("canvas")
  canvas.width = 512
  canvas.height = 512
  const ctx = canvas.getContext("2d")
  const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height)
  gradient.addColorStop(0, "#1f242c")
  gradient.addColorStop(1, "#343a43")
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, canvas.width, canvas.height)

  for (let i = 0; i < 700; i += 1) {
    ctx.fillStyle = `rgba(255,255,255,${0.012 + Math.random() * 0.02})`
    ctx.fillRect(Math.random() * canvas.width, Math.random() * canvas.height, 1, 12 + Math.random() * 20)
  }

  const map = textureFromCanvas(canvas, { repeat: [3, 3] })
  const normalMap = buildNormalMapFromCanvas(canvas, 1.4)
  normalMap.repeat.set(3, 3)
  const roughnessMap = buildRoughnessMapFromCanvas(canvas, [3, 3])
  return { map, normalMap, roughnessMap }
}

function createWoodAccentTextureSet() {
  const map = createWoodFloorTexture({
    base: "#8f6646",
    dark: "#5d3d27",
    seam: "rgba(42,22,10,0.42)",
  })
  const sourceCanvas = map.image
  const normalMap = buildNormalMapFromCanvas(sourceCanvas, 1.6)
  normalMap.repeat.copy(map.repeat)
  const roughnessMap = buildRoughnessMapFromCanvas(sourceCanvas, [map.repeat.x, map.repeat.y])
  return { map, normalMap, roughnessMap }
}

function createStoneTextureSet() {
  const canvas = document.createElement("canvas")
  canvas.width = 1024
  canvas.height = 1024
  const ctx = canvas.getContext("2d")

  ctx.fillStyle = "#80776f"
  ctx.fillRect(0, 0, canvas.width, canvas.height)

  const cols = 8
  const rows = 6
  const tileW = canvas.width / cols
  const tileH = canvas.height / rows
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      const base = 104 + Math.floor(Math.random() * 42)
      ctx.fillStyle = `rgb(${base},${base - 4},${base - 8})`
      const px = x * tileW + 4
      const py = y * tileH + 4
      ctx.fillRect(px, py, tileW - 8, tileH - 8)

      for (let i = 0; i < 80; i += 1) {
        ctx.fillStyle = `rgba(255,255,255,${0.015 + Math.random() * 0.04})`
        ctx.fillRect(
          px + Math.random() * (tileW - 12),
          py + Math.random() * (tileH - 12),
          1 + Math.random() * 5,
          1 + Math.random() * 5
        )
      }
    }
  }

  ctx.strokeStyle = "rgba(222,214,204,0.22)"
  ctx.lineWidth = 4
  for (let x = 1; x < cols; x += 1) {
    ctx.beginPath()
    ctx.moveTo(x * tileW, 0)
    ctx.lineTo(x * tileW, canvas.height)
    ctx.stroke()
  }
  for (let y = 1; y < rows; y += 1) {
    ctx.beginPath()
    ctx.moveTo(0, y * tileH)
    ctx.lineTo(canvas.width, y * tileH)
    ctx.stroke()
  }

  const map = textureFromCanvas(canvas, { repeat: [3.2, 3.2] })
  const normalMap = buildNormalMapFromCanvas(canvas, 2.35)
  normalMap.repeat.set(3.2, 3.2)
  const roughnessMap = buildRoughnessMapFromCanvas(canvas, [3.2, 3.2])
  return { map, normalMap, roughnessMap }
}

function getPlasterTextureSet() {
  if (!plasterTextureSetCache) plasterTextureSetCache = createPlasterTextureSet()
  return plasterTextureSetCache
}

function getConcreteTextureSet() {
  if (!concreteTextureSetCache) concreteTextureSetCache = createConcreteTextureSet()
  return concreteTextureSetCache
}

function getRoofTextureSet() {
  if (!roofTextureSetCache) roofTextureSetCache = createRoofTextureSet()
  return roofTextureSetCache
}

function getDarkMetalTextureSet() {
  if (!darkMetalTextureSetCache) darkMetalTextureSetCache = createDarkMetalTextureSet()
  return darkMetalTextureSetCache
}

function getWoodAccentTextureSet() {
  if (!woodTextureSetCache) woodTextureSetCache = createWoodAccentTextureSet()
  return woodTextureSetCache
}

function getStoneTextureSet() {
  if (!stoneTextureSetCache) stoneTextureSetCache = createStoneTextureSet()
  return stoneTextureSetCache
}

function createWoodFloorTexture({
  base = "#cda57c",
  dark = "#9f744f",
  seam = "rgba(70,44,24,0.28)",
} = {}) {
  const canvas = document.createElement("canvas")
  canvas.width = 1024
  canvas.height = 1024
  const ctx = canvas.getContext("2d")

  ctx.fillStyle = base
  ctx.fillRect(0, 0, canvas.width, canvas.height)

  const plankCount = 10
  const plankWidth = canvas.width / plankCount
  for (let i = 0; i < plankCount; i += 1) {
    const x = i * plankWidth
    const lightness = 54 + (i % 3) * 5 - ((i + 1) % 2) * 3
    ctx.fillStyle = `hsl(30, 40%, ${lightness}%)`
    ctx.fillRect(x, 0, plankWidth, canvas.height)

    for (let y = 0; y < canvas.height; y += 18) {
      const alpha = 0.045 + ((i + y / 18) % 5) * 0.014
      ctx.fillStyle = `rgba(92, 58, 31, ${alpha})`
      ctx.fillRect(x, y, plankWidth, 6)
    }

    for (let y = 0; y < canvas.height; y += 58) {
      ctx.fillStyle = "rgba(64,38,18,0.1)"
      ctx.fillRect(x + 6, y, plankWidth - 12, 2)
    }

    ctx.fillStyle = seam
    ctx.fillRect(x, 0, 3, canvas.height)

    for (let k = 0; k < 8; k += 1) {
      const knotX = x + 12 + Math.random() * (plankWidth - 24)
      const knotY = 30 + Math.random() * (canvas.height - 60)
      ctx.beginPath()
      ctx.fillStyle = `rgba(85,52,28,${0.06 + Math.random() * 0.06})`
      ctx.ellipse(knotX, knotY, 8 + Math.random() * 8, 3 + Math.random() * 4, Math.random(), 0, Math.PI * 2)
      ctx.fill()
    }

    ctx.fillStyle = dark
    ctx.fillRect(x + plankWidth - 3, 0, 3, canvas.height)
  }

  for (let i = 0; i < 220; i += 1) {
    ctx.fillStyle = `rgba(255,255,255,${0.015 + (i % 6) * 0.005})`
    ctx.fillRect(
      Math.random() * canvas.width,
      Math.random() * canvas.height,
      24 + Math.random() * 42,
      2
    )
  }

  for (let i = 0; i < 160; i += 1) {
    ctx.fillStyle = `rgba(70,42,24,${0.02 + (i % 5) * 0.01})`
    ctx.fillRect(
      Math.random() * canvas.width,
      Math.random() * canvas.height,
      18 + Math.random() * 30,
      1.5
    )
  }

  const texture = new THREE.CanvasTexture(canvas)
  texture.wrapS = THREE.RepeatWrapping
  texture.wrapT = THREE.RepeatWrapping
  texture.repeat.set(4.6, 4.6)
  texture.anisotropy = 8
  texture.colorSpace = THREE.SRGBColorSpace
  return texture
}

function createSkyTexture() {
  const canvas = document.createElement("canvas")
  canvas.width = 1024
  canvas.height = 512
  const ctx = canvas.getContext("2d")

  const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height)
  gradient.addColorStop(0, "#7e9fbc")
  gradient.addColorStop(0.34, "#b7cbdb")
  gradient.addColorStop(0.62, "#edf1ef")
  gradient.addColorStop(1, "#f2e6d8")
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, canvas.width, canvas.height)

  const sunGlow = ctx.createRadialGradient(canvas.width * 0.72, canvas.height * 0.18, 10, canvas.width * 0.72, canvas.height * 0.18, 180)
  sunGlow.addColorStop(0, "rgba(255,242,214,0.85)")
  sunGlow.addColorStop(0.35, "rgba(255,235,200,0.28)")
  sunGlow.addColorStop(1, "rgba(255,235,200,0)")
  ctx.fillStyle = sunGlow
  ctx.fillRect(0, 0, canvas.width, canvas.height)

  for (let i = 0; i < 14; i += 1) {
    const x = Math.random() * canvas.width
    const y = 40 + Math.random() * 180
    const w = 120 + Math.random() * 180
    const h = 30 + Math.random() * 40
    const cloud = ctx.createRadialGradient(x, y, 10, x, y, w)
    cloud.addColorStop(0, "rgba(255,255,255,0.42)")
    cloud.addColorStop(0.45, "rgba(255,255,255,0.16)")
    cloud.addColorStop(1, "rgba(255,255,255,0)")
    ctx.fillStyle = cloud
    ctx.beginPath()
    ctx.ellipse(x, y, w, h, 0, 0, Math.PI * 2)
    ctx.fill()
  }

  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  return texture
}

function createGroundTexture() {
  const canvas = document.createElement("canvas")
  canvas.width = 1024
  canvas.height = 1024
  const ctx = canvas.getContext("2d")

  ctx.fillStyle = "#d9d3cb"
  ctx.fillRect(0, 0, canvas.width, canvas.height)

  for (let i = 0; i < 1800; i += 1) {
    const alpha = 0.02 + Math.random() * 0.045
    const gray = 188 + Math.floor(Math.random() * 22)
    ctx.fillStyle = `rgba(${gray},${gray - 4},${gray - 8},${alpha})`
    const size = 2 + Math.random() * 7
    ctx.fillRect(Math.random() * canvas.width, Math.random() * canvas.height, size, size)
  }

  for (let i = 0; i < 70; i += 1) {
    ctx.strokeStyle = `rgba(164,156,146,${0.04 + Math.random() * 0.04})`
    ctx.lineWidth = 1 + Math.random() * 2
    ctx.beginPath()
    ctx.moveTo(Math.random() * canvas.width, Math.random() * canvas.height)
    ctx.lineTo(Math.random() * canvas.width, Math.random() * canvas.height)
    ctx.stroke()
  }

  const texture = new THREE.CanvasTexture(canvas)
  texture.wrapS = THREE.RepeatWrapping
  texture.wrapT = THREE.RepeatWrapping
  texture.repeat.set(14, 14)
  texture.anisotropy = 8
  texture.colorSpace = THREE.SRGBColorSpace
  return texture
}

function getInteriorFloorMaterial(floorIndex) {
  if (!interiorFloorMaterialCache) {
    const woodSet = getWoodAccentTextureSet()
    interiorFloorMaterialCache = new THREE.MeshStandardMaterial({
      map: woodSet.map,
      normalMap: woodSet.normalMap,
      roughnessMap: woodSet.roughnessMap,
      color: "#f0dcc2",
      roughness: 0.48,
      metalness: 0.02,
    })
  }
  return interiorFloorMaterialCache
}

function createInteriorFloorFinish({ shape, floorIndex, yBase }) {
  const finishGeo = new THREE.ExtrudeGeometry(shape, {
    depth: 0.016,
    bevelEnabled: false,
    curveSegments: 1,
  })
  finishGeo.rotateX(-Math.PI / 2)

  const finish = new THREE.Mesh(finishGeo, getInteriorFloorMaterial(floorIndex))
  finish.position.y = yBase + 0.006
  finish.material.polygonOffset = true
  finish.material.polygonOffsetFactor = -1
  finish.material.polygonOffsetUnits = -1
  finish.receiveShadow = true
  return finish
}

function buildProjectGroup(project, geometryMeta, activeFloor, floorHeightMeters) {
  const group = new THREE.Group()
  group.name = "Plan2MassModel"

  const floors = project?.floors || []
  const renderAllFloors = activeFloor === "building"
  const plasterSet = getPlasterTextureSet()
  const concreteSet = getConcreteTextureSet()
  const roofSet = getRoofTextureSet()
  const metalSet = getDarkMetalTextureSet()
  const woodSet = getWoodAccentTextureSet()

  const outerWallMap = plasterSet.map.clone()
  const outerWallNormal = plasterSet.normalMap.clone()
  const outerWallRoughness = plasterSet.roughnessMap.clone()
  outerWallMap.repeat.set(4, 4)
  outerWallNormal.repeat.set(4, 4)
  outerWallRoughness.repeat.set(4, 4)
  const outerWallMaterial = new THREE.MeshStandardMaterial({
    map: outerWallMap,
    normalMap: outerWallNormal,
    roughnessMap: outerWallRoughness,
    color: "#e7e0d5",
    roughness: 0.9,
    metalness: 0.01,
  })

  const innerWallMaterial = new THREE.MeshStandardMaterial({
    map: plasterSet.map.clone(),
    normalMap: plasterSet.normalMap.clone(),
    roughnessMap: plasterSet.roughnessMap.clone(),
    color: "#efe8de",
    roughness: 0.92,
    metalness: 0.01,
  })
  innerWallMaterial.map.repeat.set(4, 4)
  innerWallMaterial.normalMap.repeat.set(4, 4)
  innerWallMaterial.roughnessMap.repeat.set(4, 4)

  const slabMap = concreteSet.map.clone()
  const slabNormal = concreteSet.normalMap.clone()
  const slabRoughness = concreteSet.roughnessMap.clone()
  slabMap.repeat.set(8, 8)
  slabNormal.repeat.set(8, 8)
  slabRoughness.repeat.set(8, 8)
  const slabMaterial = new THREE.MeshStandardMaterial({
    map: slabMap,
    normalMap: slabNormal,
    roughnessMap: slabRoughness,
    color: "#ddd5ca",
    roughness: 0.9,
    metalness: 0.0,
  })

  const roofMaterial = new THREE.MeshStandardMaterial({
    map: roofSet.map,
    normalMap: roofSet.normalMap,
    roughnessMap: roofSet.roughnessMap,
    color: "#837968",
    roughness: 0.95,
    metalness: 0.0,
  })

  const doorMaterial = new THREE.MeshStandardMaterial({
    map: woodSet.map,
    normalMap: woodSet.normalMap,
    roughnessMap: woodSet.roughnessMap,
    color: "#74503a",
    roughness: 0.5,
    metalness: 0.05,
  })

  const frameMaterial = new THREE.MeshStandardMaterial({
    map: metalSet.map,
    normalMap: metalSet.normalMap,
    roughnessMap: metalSet.roughnessMap,
    color: "#f2eee7",
    roughness: 0.42,
    metalness: 0.18,
  })

  const glassMaterial = new THREE.MeshPhysicalMaterial({
    color: "#dcebF6",
    roughness: 0,
    metalness: 0.0,
    transparent: true,
    opacity: 0.4,
    transmission: 0.98,
    thickness: 0.32,
    ior: 1.5,
    envMap: sceneHdriTextureCache || sceneSkyTextureCache,
    envMapIntensity: 1.8,
  })

  const lineMaterial = new THREE.LineBasicMaterial({
    color: "#8f867a",
  })

  if (renderAllFloors) {
    const buildingOuterWallMaterial = new THREE.MeshStandardMaterial({
      map: plasterSet.map.clone(),
      normalMap: plasterSet.normalMap.clone(),
      roughnessMap: plasterSet.roughnessMap.clone(),
      color: "#e6e0d7",
      roughness: 0.86,
      metalness: 0.01,
    })
    buildingOuterWallMaterial.map.repeat.set(6, 6)
    buildingOuterWallMaterial.normalMap.repeat.set(6, 6)
    buildingOuterWallMaterial.roughnessMap.repeat.set(6, 6)

    const buildingInnerWallMaterial = new THREE.MeshStandardMaterial({
      map: plasterSet.map.clone(),
      normalMap: plasterSet.normalMap.clone(),
      roughnessMap: plasterSet.roughnessMap.clone(),
      color: "#f7f5f0",
      roughness: 0.9,
      metalness: 0.0,
    })
    buildingInnerWallMaterial.map.repeat.set(4.5, 4.5)
    buildingInnerWallMaterial.normalMap.repeat.set(4.5, 4.5)
    buildingInnerWallMaterial.roughnessMap.repeat.set(4.5, 4.5)

    const buildingSlabMaterial = new THREE.MeshStandardMaterial({
      map: concreteSet.map.clone(),
      normalMap: concreteSet.normalMap.clone(),
      roughnessMap: concreteSet.roughnessMap.clone(),
      color: "#d8d0c5",
      roughness: 0.74,
      metalness: 0.04,
    })

    const buildingRoofMaterial = new THREE.MeshStandardMaterial({
      map: roofSet.map.clone(),
      normalMap: roofSet.normalMap.clone(),
      roughnessMap: roofSet.roughnessMap.clone(),
      color: "#272c33",
      roughness: 0.68,
      metalness: 0.08,
    })

    const buildingDoorMaterial = new THREE.MeshStandardMaterial({
      map: woodSet.map.clone(),
      normalMap: woodSet.normalMap.clone(),
      roughnessMap: woodSet.roughnessMap.clone(),
      color: "#5a3c2d",
      roughness: 0.34,
      metalness: 0.1,
    })

    const buildingFrameMaterial = new THREE.MeshStandardMaterial({
      map: metalSet.map.clone(),
      normalMap: metalSet.normalMap.clone(),
      roughnessMap: metalSet.roughnessMap.clone(),
      color: "#191d21",
      roughness: 0.28,
      metalness: 0.5,
    })

    const buildingGlassMaterial = new THREE.MeshPhysicalMaterial({
      color: "#d8edf8",
      roughness: 0.03,
      metalness: 0.0,
      transparent: true,
      opacity: 0.62,
      transmission: 0.97,
      thickness: 0.34,
      ior: 1.46,
      envMap: sceneSkyTextureCache,
      envMapIntensity: 1.7,
    })

    return buildBuildingExteriorGroup({
      project,
      geometryMeta,
      floorHeightMeters,
      materials: {
        outerWallMaterial: buildingOuterWallMaterial,
        innerWallMaterial: buildingInnerWallMaterial,
        slabMaterial: buildingSlabMaterial,
        roofMaterial: buildingRoofMaterial,
        doorMaterial: buildingDoorMaterial,
        frameMaterial: buildingFrameMaterial,
        glassMaterial: buildingGlassMaterial,
      },
    })
  }

  const yStep = renderAllFloors ? floorHeightMeters + BUILDING_FLOOR_GAP : floorHeightMeters

  floors.forEach((floor, index) => {
    if (!renderAllFloors && activeFloor !== index) return

    const floorGroup = buildSingleFloorGroup({
      floor,
      floorIndex: index,
      totalFloors: floors.length,
      offsetY: renderAllFloors ? index * yStep : 0,
      floorHeight: floorHeightMeters,
      geometryMeta,
      materials: {
        outerWallMaterial,
        innerWallMaterial,
        slabMaterial,
        roofMaterial,
        doorMaterial,
        frameMaterial,
        glassMaterial,
        lineMaterial,
      },
      includeRoof: false,
    })

    group.add(floorGroup)
  })

  return group
}

function buildBuildingExteriorGroup({
  project,
  geometryMeta,
  floorHeightMeters,
  materials,
}) {
  const group = new THREE.Group()
  group.name = "Plan2MassBuilding"

  const floors = project?.floors || []
  const totalHeight = floors.length * floorHeightMeters
  const envelope = getBuildingEnvelope(floors[0]?.polygon || [], geometryMeta)

  if (envelope) {
    const siteContext = createBuildingSiteContext(envelope, totalHeight)
    group.add(siteContext)
  }

  floors.forEach((floor, index) => {
    const floorGroup = createBuildingFloorShell({
      floor,
      floorIndex: index,
      totalFloors: floors.length,
      offsetY: index * floorHeightMeters,
      floorHeight: floorHeightMeters,
      geometryMeta,
      materials: {
        outerWallMaterial: materials.outerWallMaterial,
        innerWallMaterial: materials.innerWallMaterial || materials.outerWallMaterial,
        slabMaterial: materials.slabMaterial,
        roofMaterial: materials.roofMaterial,
        doorMaterial: materials.doorMaterial,
        frameMaterial: materials.frameMaterial,
        glassMaterial: materials.glassMaterial,
      },
    })
    group.add(floorGroup)
  })

  if (envelope) {
    const facadeDetails = createFacadeArticulation({
      envelope,
      totalHeight,
      floorHeightMeters,
      floors,
      geometryMeta,
    })
    group.add(facadeDetails)
  }

  const roof = createPitchedRoof({
    polygon: floors[0]?.polygon || [],
    geometryMeta,
    yBase: totalHeight + SLAB_THICKNESS,
    material: materials.roofMaterial,
  })
  if (roof) group.add(roof)

  const entrance = createEntranceFeature({
    floor: floors[0],
    geometryMeta,
    yBase: SLAB_THICKNESS,
    material: materials.doorMaterial,
  })
  if (entrance) group.add(entrance)

  return group
}

function createBuildingFloorShell({
  floor,
  floorIndex,
  totalFloors,
  offsetY,
  floorHeight,
  geometryMeta,
  materials,
}) {
  const group = new THREE.Group()
  const polygon = floor?.polygon || []
  if (polygon.length < 3) return group

  const transformedPolygon = polygon.map(([x, y]) => geometryMeta.transformPoint(x, y))
  const shape = polygonToShape(transformedPolygon)

  const slabGeo = new THREE.ExtrudeGeometry(shape, {
    depth: SLAB_THICKNESS,
    bevelEnabled: false,
    curveSegments: 1,
  })
  slabGeo.rotateX(-Math.PI / 2)

  const slab = new THREE.Mesh(slabGeo, materials.slabMaterial)
  slab.position.y = offsetY
  slab.castShadow = true
  slab.receiveShadow = true
  group.add(slab)

  const wallGraph = buildWallGraph({
    polygon,
    innerWalls: floor.inner_walls || [],
  })

  const classifiedOpenings = classifyOpenings({
    graph: wallGraph,
    doors: floor.doors || [],
    windows: floor.windows || [],
    doorWidthPx: DOOR_OPENING_WIDTH_PX,
    windowWidthPx: WINDOW_OPENING_WIDTH_PX,
  })

  const { matched } = matchOpeningsToWalls({
    graph: wallGraph,
    openings: classifiedOpenings,
  })

  const exteriorOpeningsByWallId = new Map()
  matched
    .filter((opening) => opening.wallKind === "outer")
    .forEach((opening) => {
      if (!exteriorOpeningsByWallId.has(opening.wallId)) {
        exteriorOpeningsByWallId.set(opening.wallId, [])
      }
      exteriorOpeningsByWallId.get(opening.wallId).push(opening)
    })

  wallGraph.walls
    .filter((wall) => wall.kind === "outer")
    .forEach((wall) => {
      const wallMesh = createWallSegmentMesh({
        line: wall.line,
        geometryMeta,
        height: floorHeight,
        thickness: OUTER_WALL_THICKNESS,
        material: materials.outerWallMaterial,
        yBase: offsetY + SLAB_THICKNESS,
        openings: exteriorOpeningsByWallId.get(wall.id) || [],
      })
      if (wallMesh) group.add(wallMesh)
    })

  matched
    .filter((opening) => opening.wallKind === "outer")
    .forEach((opening) => {
      if (opening.type === "window") {
        const windowInsert = createWindowInsert({
          point: opening.point,
          line: opening.hostLine,
          geometryMeta,
          widthPx: opening.widthPx,
          height: WINDOW_HEIGHT,
          sillHeight: WINDOW_SILL_HEIGHT,
          thickness: OUTER_WALL_THICKNESS,
          yBase: offsetY + SLAB_THICKNESS,
          frameMaterial: materials.frameMaterial,
          glassMaterial: materials.glassMaterial,
        })
        if (windowInsert) group.add(windowInsert)
        return
      }

      if (opening.type === "door") {
        const doorInsert = createDoorInsert({
          point: opening.point,
          line: opening.hostLine,
          geometryMeta,
          wallKind: "outer",
          widthPx: opening.widthPx,
          height: floorHeight * DOOR_HEIGHT_RATIO,
          thickness: OUTER_WALL_THICKNESS,
          yBase: offsetY + SLAB_THICKNESS,
          material: materials.doorMaterial,
          frameMaterial: materials.frameMaterial,
        })
        if (doorInsert) group.add(doorInsert)
      }
    })

  const slabShadow = new THREE.Mesh(
    new THREE.ExtrudeGeometry(shape, {
      depth: 0.04,
      bevelEnabled: false,
      curveSegments: 1,
    }).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({
      color: floorIndex === totalFloors - 1 ? "#b6ad9f" : "#b0a695",
      roughness: 0.96,
      metalness: 0.0,
    })
  )
  slabShadow.position.y = offsetY + floorHeight + SLAB_THICKNESS - 0.04
  slabShadow.receiveShadow = true
  group.add(slabShadow)

  return group
}

function getBuildingEnvelope(polygon, geometryMeta) {
  if (!polygon?.length) return null
  const pts = polygon.map(([x, y]) => geometryMeta.transformPoint(x, y))
  const xs = pts.map((p) => p.x)
  const zs = pts.map((p) => p.z)
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minZ: Math.min(...zs),
    maxZ: Math.max(...zs),
  }
}

function createBuildingSiteContext(envelope, totalHeight) {
  const group = new THREE.Group()
  const width = envelope.maxX - envelope.minX
  const depth = envelope.maxZ - envelope.minZ
  const centerX = (envelope.minX + envelope.maxX) / 2
  const centerZ = (envelope.minZ + envelope.maxZ) / 2
  const siteMinX = centerX - Math.max(width * 0.72, 10.6)
  const siteMaxX = centerX + Math.max(width * 0.72, 10.6)
  const siteMinZ = centerZ - Math.max(depth * 0.54, 7.6)
  const siteMaxZ = envelope.maxZ + 3.4
  const gateWidth = Math.max(4.4, width * 0.3)
  const concreteSet = getConcreteTextureSet()
  const stoneSet = getStoneTextureSet()
  const darkMetalSet = getDarkMetalTextureSet()

  const plinthMat = new THREE.MeshStandardMaterial({
    map: concreteSet.map.clone(),
    normalMap: concreteSet.normalMap.clone(),
    roughnessMap: concreteSet.roughnessMap.clone(),
    color: "#c9c1b5",
    roughness: 0.96,
    metalness: 0.0,
  })
  const plinth = new THREE.Mesh(
    new THREE.BoxGeometry(width + 1.4, 0.34, depth + 1.2),
    plinthMat
  )
  plinth.position.set(centerX, 0.02, centerZ)
  plinth.receiveShadow = true
  group.add(plinth)

  const forecourtMat = new THREE.MeshStandardMaterial({
    map: concreteSet.map.clone(),
    normalMap: concreteSet.normalMap.clone(),
    roughnessMap: concreteSet.roughnessMap.clone(),
    color: "#ddd7cf",
    roughness: 0.92,
    metalness: 0.0,
  })
  const forecourt = new THREE.Mesh(
    new THREE.BoxGeometry(Math.max(width + 2.2, width * 1.08), 0.05, Math.max(4.1, depth * 0.26)),
    forecourtMat
  )
  forecourt.position.set(centerX, 0.22, envelope.maxZ + 1.32)
  forecourt.receiveShadow = true
  group.add(forecourt)

  const path = new THREE.Mesh(
    new THREE.BoxGeometry(Math.max(2.2, width * 0.14), 0.05, 3.2),
    new THREE.MeshStandardMaterial({
      color: "#d3c9ba",
      roughness: 0.86,
      metalness: 0.0,
    })
  )
  path.position.set(centerX, 0.24, envelope.maxZ + 0.98)
  path.receiveShadow = true
  group.add(path)

  const apron = new THREE.Mesh(
    new THREE.BoxGeometry(width + 0.24, 0.03, 0.72),
    new THREE.MeshStandardMaterial({
      color: "#e7e1d8",
      roughness: 0.82,
      metalness: 0.0,
    })
  )
  apron.position.set(centerX, 0.21, envelope.maxZ + 0.34)
  apron.receiveShadow = true
  group.add(apron)

  const sideWalkMaterial = new THREE.MeshStandardMaterial({
    map: concreteSet.map.clone(),
    normalMap: concreteSet.normalMap.clone(),
    roughnessMap: concreteSet.roughnessMap.clone(),
    color: "#d2cabf",
    roughness: 0.9,
    metalness: 0.0,
  })
  ;[-1, 1].forEach((side) => {
    const walk = new THREE.Mesh(
      new THREE.BoxGeometry(0.52, 0.03, depth + 0.84),
      sideWalkMaterial
    )
    walk.position.set(centerX + side * (width / 2 + 0.48), 0.2, centerZ)
    walk.receiveShadow = true
    group.add(walk)
  })

  const driveway = new THREE.Mesh(
    new THREE.BoxGeometry(gateWidth + 0.8, 0.04, 3.8),
    new THREE.MeshStandardMaterial({
      map: concreteSet.map.clone(),
      normalMap: concreteSet.normalMap.clone(),
      roughnessMap: concreteSet.roughnessMap.clone(),
      color: "#d7d0c6",
      roughness: 0.84,
      metalness: 0.0,
    })
  )
  driveway.position.set(centerX, 0.22, siteMaxZ - 1.3)
  driveway.receiveShadow = true
  group.add(driveway)

  const stoneWallMat = new THREE.MeshStandardMaterial({
    map: stoneSet.map.clone(),
    normalMap: stoneSet.normalMap.clone(),
    roughnessMap: stoneSet.roughnessMap.clone(),
    color: "#6d665f",
    roughness: 0.97,
    metalness: 0.0,
  })
  const capMat = new THREE.MeshStandardMaterial({
    map: concreteSet.map.clone(),
    normalMap: concreteSet.normalMap.clone(),
    roughnessMap: concreteSet.roughnessMap.clone(),
    color: "#ece6dc",
    roughness: 0.78,
    metalness: 0.02,
  })
  const metalGateMat = new THREE.MeshStandardMaterial({
    map: darkMetalSet.map.clone(),
    normalMap: darkMetalSet.normalMap.clone(),
    roughnessMap: darkMetalSet.roughnessMap.clone(),
    color: "#14181d",
    roughness: 0.42,
    metalness: 0.54,
  })

  const wallHeight = 0.88
  const wallThickness = 0.14

  const addPerimeterWall = (w, h, d, x, y, z, material) => {
    const wall = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material)
    wall.position.set(x, y, z)
    wall.castShadow = true
    wall.receiveShadow = true
    group.add(wall)
    return wall
  }

  addPerimeterWall(
    siteMaxX - siteMinX,
    0.08,
    wallThickness + 0.04,
    centerX,
    wallHeight + 0.06,
    siteMinZ,
    capMat
  )
  addPerimeterWall(
    siteMaxX - siteMinX,
    wallHeight,
    wallThickness,
    centerX,
    wallHeight / 2,
    siteMinZ,
    stoneWallMat
  )

  addPerimeterWall(
    wallThickness,
    wallHeight,
    siteMaxZ - siteMinZ,
    siteMinX,
    wallHeight / 2,
    centerZ + (siteMaxZ - siteMinZ) * 0.1,
    stoneWallMat
  )
  addPerimeterWall(
    wallThickness,
    wallHeight,
    siteMaxZ - siteMinZ,
    siteMaxX,
    wallHeight / 2,
    centerZ + (siteMaxZ - siteMinZ) * 0.1,
    stoneWallMat
  )
  addPerimeterWall(
    wallThickness + 0.04,
    0.08,
    siteMaxZ - siteMinZ,
    siteMinX,
    wallHeight + 0.06,
    centerZ + (siteMaxZ - siteMinZ) * 0.1,
    capMat
  )
  addPerimeterWall(
    wallThickness + 0.04,
    0.08,
    siteMaxZ - siteMinZ,
    siteMaxX,
    wallHeight + 0.06,
    centerZ + (siteMaxZ - siteMinZ) * 0.1,
    capMat
  )

  const frontWallLeftWidth = (siteMaxX - siteMinX - gateWidth) / 2 - 0.5
  const frontWallY = wallHeight / 2
  const frontCapY = wallHeight + 0.06
  const frontWallZ = siteMaxZ

  const frontLeftX = siteMinX + frontWallLeftWidth / 2
  const frontRightX = siteMaxX - frontWallLeftWidth / 2

  ;[
    [frontLeftX, frontWallLeftWidth],
    [frontRightX, frontWallLeftWidth],
  ].forEach(([x, w]) => {
    addPerimeterWall(w, wallHeight, wallThickness, x, frontWallY, frontWallZ, stoneWallMat)
    addPerimeterWall(w + 0.04, 0.08, wallThickness + 0.04, x, frontCapY, frontWallZ, capMat)
  })

  ;[-1, 1].forEach((side) => {
    const pier = new THREE.Mesh(
      new THREE.BoxGeometry(0.24, 1.18, 0.24),
      capMat
    )
    pier.position.set(centerX + side * (gateWidth / 2 + 0.12), 0.59, frontWallZ)
    pier.castShadow = true
    pier.receiveShadow = true
    group.add(pier)

  })

  const gate = new THREE.Mesh(
    new THREE.BoxGeometry(gateWidth, 0.92, 0.07),
    metalGateMat
  )
  gate.position.set(centerX, 0.46, frontWallZ)
  gate.castShadow = true
  gate.receiveShadow = true
  group.add(gate)

  for (let i = -3; i <= 3; i += 1) {
    const slat = new THREE.Mesh(
      new THREE.BoxGeometry(0.03, 0.84, 0.03),
      new THREE.MeshStandardMaterial({
        color: "#2a2f34",
        roughness: 0.42,
        metalness: 0.48,
      })
    )
    slat.position.set(centerX + i * (gateWidth / 8), 0.46, frontWallZ + 0.025)
    group.add(slat)
  }

  const backdropShadow = new THREE.Mesh(
    new THREE.CircleGeometry(Math.max(width, depth) * 1.38, 48),
    new THREE.MeshBasicMaterial({
      color: "#9d9487",
      transparent: true,
      opacity: 0.12,
      depthWrite: false,
    })
  )
  backdropShadow.rotation.x = -Math.PI / 2
  backdropShadow.position.set(centerX + 1.2, 0.03, centerZ + 0.4)
  group.add(backdropShadow)

  return group
}

function collectExteriorOpeningsByFace(floors, geometryMeta, envelope) {
  const faces = {
    front: [],
    back: [],
    left: [],
    right: [],
  }

  floors.forEach((floor, floorIndex) => {
    ;["windows", "doors"].forEach((key) => {
      ;(floor[key] || []).forEach((opening) => {
        const world = geometryMeta.transformPoint(opening.x, opening.y)
        const distances = {
          front: Math.abs(world.z - envelope.maxZ),
          back: Math.abs(world.z - envelope.minZ),
          left: Math.abs(world.x - envelope.minX),
          right: Math.abs(world.x - envelope.maxX),
        }
        const face = Object.entries(distances).sort((a, b) => a[1] - b[1])[0]?.[0]
        if (!face) return
        faces[face].push({
          ...opening,
          type: key === "doors" ? "door" : "window",
          floorIndex,
          world,
        })
      })
    })
  })

  return faces
}

function createFacadeArticulation({ envelope, totalHeight, floorHeightMeters, floors, geometryMeta }) {
  const group = new THREE.Group()
  const width = envelope.maxX - envelope.minX
  const depth = envelope.maxZ - envelope.minZ
  const centerX = (envelope.minX + envelope.maxX) / 2
  const centerZ = (envelope.minZ + envelope.maxZ) / 2
  const concreteSet = getConcreteTextureSet()
  const stoneSet = getStoneTextureSet()
  const woodSet = getWoodAccentTextureSet()
  const metalSet = getDarkMetalTextureSet()

  const accentMat = new THREE.MeshStandardMaterial({
    map: concreteSet.map.clone(),
    normalMap: concreteSet.normalMap.clone(),
    roughnessMap: concreteSet.roughnessMap.clone(),
    color: "#ded5c8",
    roughness: 0.82,
    metalness: 0.02,
  })
  const trimMat = new THREE.MeshStandardMaterial({
    map: concreteSet.map.clone(),
    normalMap: concreteSet.normalMap.clone(),
    roughnessMap: concreteSet.roughnessMap.clone(),
    color: "#c8beb1",
    roughness: 0.88,
    metalness: 0.0,
  })
  const shadowMat = new THREE.MeshStandardMaterial({
    map: concreteSet.map.clone(),
    normalMap: concreteSet.normalMap.clone(),
    roughnessMap: concreteSet.roughnessMap.clone(),
    color: "#b4a697",
    roughness: 0.94,
    metalness: 0.0,
  })
  const lightTrimMat = new THREE.MeshStandardMaterial({
    map: concreteSet.map.clone(),
    normalMap: concreteSet.normalMap.clone(),
    roughnessMap: concreteSet.roughnessMap.clone(),
    color: "#f1ece5",
    roughness: 0.78,
    metalness: 0.02,
  })
  const woodMat = new THREE.MeshStandardMaterial({
    map: woodSet.map.clone(),
    normalMap: woodSet.normalMap.clone(),
    roughnessMap: woodSet.roughnessMap.clone(),
    color: "#6d4a34",
    roughness: 0.56,
    metalness: 0.06,
  })
  const stoneMat = new THREE.MeshStandardMaterial({
    map: stoneSet.map.clone(),
    normalMap: stoneSet.normalMap.clone(),
    roughnessMap: stoneSet.roughnessMap.clone(),
    color: "#7b746b",
    roughness: 0.95,
    metalness: 0.0,
  })
  const darkMetalMat = new THREE.MeshStandardMaterial({
    map: metalSet.map.clone(),
    normalMap: metalSet.normalMap.clone(),
    roughnessMap: metalSet.roughnessMap.clone(),
    color: "#413a34",
    roughness: 0.4,
    metalness: 0.18,
  })

  const frontZ = envelope.maxZ + 0.08
  const backZ = envelope.minZ - 0.08
  const faces = geometryMeta ? collectExteriorOpeningsByFace(floors, geometryMeta, envelope) : null

  const basePlinthHeight = 0.54
  const plinth = new THREE.Mesh(
    new THREE.BoxGeometry(width + 0.28, basePlinthHeight, depth + 0.28),
    stoneMat
  )
  plinth.position.set(centerX, basePlinthHeight / 2 + 0.02, centerZ)
  group.add(plinth)

  const frontEntryMass = new THREE.Mesh(
    new THREE.BoxGeometry(Math.max(2.6, width * 0.22), totalHeight - 0.42, 0.08),
    accentMat
  )
  frontEntryMass.position.set(centerX, totalHeight / 2 + 0.02, frontZ + 0.01)
  group.add(frontEntryMass)

  const rearSpine = new THREE.Mesh(
    new THREE.BoxGeometry(Math.max(1.6, width * 0.14), totalHeight - 0.56, 0.06),
    shadowMat
  )
  rearSpine.position.set(centerX, totalHeight / 2, backZ - 0.015)
  group.add(rearSpine)

  const cornerDepth = 0.11
  const cornerWidth = 0.18
  ;[
    [envelope.minX - 0.04, envelope.minZ],
    [envelope.minX - 0.04, envelope.maxZ],
    [envelope.maxX + 0.04, envelope.minZ],
    [envelope.maxX + 0.04, envelope.maxZ],
  ].forEach(([x, z]) => {
    const corner = new THREE.Mesh(
      new THREE.BoxGeometry(cornerWidth, totalHeight + 0.1, cornerDepth),
      trimMat
    )
    corner.position.set(x, totalHeight / 2 + 0.05, z)
    group.add(corner)
  })

  for (let i = 1; i < floors.length; i += 1) {
    const y = i * floorHeightMeters + 0.08
    const frontBelt = new THREE.Mesh(
      new THREE.BoxGeometry(width + 0.08, 0.04, 0.1),
      trimMat
    )
    frontBelt.position.set(centerX, y, frontZ + 0.02)
    group.add(frontBelt)

    const backBelt = new THREE.Mesh(
      new THREE.BoxGeometry(width + 0.04, 0.035, 0.08),
      shadowMat
    )
    backBelt.position.set(centerX, y, backZ - 0.02)
    group.add(backBelt)

    const leftBelt = new THREE.Mesh(
      new THREE.BoxGeometry(0.08, 0.04, depth + 0.04),
      trimMat
    )
    leftBelt.position.set(envelope.minX - 0.02, y, centerZ)
    group.add(leftBelt)

    const rightBelt = new THREE.Mesh(
      new THREE.BoxGeometry(0.08, 0.04, depth + 0.04),
      trimMat
    )
    rightBelt.position.set(envelope.maxX + 0.02, y, centerZ)
    group.add(rightBelt)
  }

  const crown = new THREE.Mesh(
    new THREE.BoxGeometry(width + 0.22, 0.09, depth + 0.22),
    trimMat
  )
  crown.position.set(centerX, totalHeight + 0.05, centerZ)
  group.add(crown)

  if (faces) {
    const addOpeningFrame = ({ axis, opening, depthOffset = 0.06 }) => {
      const widthWorld = getOpeningWidthWorld(opening, geometryMeta.scale)
      const heightWorld = opening.type === "door" ? floorHeightMeters * DOOR_HEIGHT_RATIO : WINDOW_HEIGHT
      const baseY = opening.type === "door" ? 0 : WINDOW_SILL_HEIGHT
      const y = opening.floorIndex * floorHeightMeters + baseY + heightWorld / 2 + SLAB_THICKNESS

      if (axis === "front" || axis === "back") {
        const z = axis === "front" ? frontZ + depthOffset : backZ - depthOffset
        const surround = new THREE.Mesh(
          new THREE.BoxGeometry(widthWorld + 0.18, heightWorld + 0.16, 0.08),
          lightTrimMat
        )
        surround.position.set(opening.world.x, y, z)
        group.add(surround)

        const head = new THREE.Mesh(
          new THREE.BoxGeometry(widthWorld + 0.16, 0.04, 0.12),
          shadowMat
        )
        head.position.set(opening.world.x, y + heightWorld / 2 + 0.07, z + (axis === "front" ? 0.015 : -0.015))
        group.add(head)

        if (opening.type === "window") {
          const sill = new THREE.Mesh(
            new THREE.BoxGeometry(widthWorld + 0.12, 0.04, 0.16),
            trimMat
          )
          sill.position.set(opening.world.x, y - heightWorld / 2 - 0.04, z + (axis === "front" ? 0.02 : -0.02))
          group.add(sill)
        }
      } else {
        const x = axis === "right" ? envelope.maxX + depthOffset : envelope.minX - depthOffset
        const surround = new THREE.Mesh(
          new THREE.BoxGeometry(0.08, heightWorld + 0.14, widthWorld + 0.18),
          lightTrimMat
        )
        surround.position.set(x, y, opening.world.z)
        group.add(surround)
      }
    }

    faces.front
      .filter((opening) => opening.type === "window")
      .forEach((opening) => addOpeningFrame({ axis: "front", opening }))
    faces.right
      .filter((opening) => opening.type === "window")
      .forEach((opening) => addOpeningFrame({ axis: "right", opening }))
    faces.left
      .filter((opening) => opening.type === "window")
      .forEach((opening) => addOpeningFrame({ axis: "left", opening }))

    const floor0 = floors[0]
    const polygon = floor0?.polygon || []
    const bottomY = polygon.length ? Math.max(...polygon.map((pt) => pt[1])) : 0
    const entry = (floor0?.doors || [])
      .filter((door) => Math.abs(door.y - bottomY) < 28)
      .sort((a, b) => (b.width || 0) - (a.width || 0))[0]
    const entryWorld = entry ? geometryMeta.transformPoint(entry.x, entry.y) : { x: centerX }
    const entryBayWidth = Math.max(2.8, width * 0.18)

    const entryBay = new THREE.Mesh(
      new THREE.BoxGeometry(entryBayWidth, floorHeightMeters + 0.56, 0.14),
      accentMat
    )
    entryBay.position.set(entryWorld.x, floorHeightMeters / 2 + 0.28, frontZ + 0.08)
    group.add(entryBay)

    const entryBayInset = new THREE.Mesh(
      new THREE.BoxGeometry(entryBayWidth - 0.26, floorHeightMeters + 0.04, 0.05),
      shadowMat
    )
    entryBayInset.position.set(entryWorld.x, floorHeightMeters / 2 + 0.14, frontZ + 0.15)
    group.add(entryBayInset)

    const woodAccent = new THREE.Mesh(
      new THREE.BoxGeometry(0.14, totalHeight - 1.02, 0.05),
      woodMat
    )
    woodAccent.position.set(entryWorld.x + entryBayWidth * 0.46, totalHeight / 2 - 0.06, frontZ + 0.09)
    group.add(woodAccent)

    const stoneAccent = new THREE.Mesh(
      new THREE.BoxGeometry(Math.max(0.7, entryBayWidth * 0.22), floorHeightMeters + 0.02, 0.06),
      stoneMat
    )
    stoneAccent.position.set(entryWorld.x - entryBayWidth * 0.48, floorHeightMeters / 2 + 0.04, frontZ + 0.1)
    group.add(stoneAccent)

    const entranceCanopyShadow = new THREE.Mesh(
      new THREE.BoxGeometry(entryBayWidth + 0.08, 0.06, 0.16),
      trimMat
    )
    entranceCanopyShadow.position.set(entryWorld.x, floorHeightMeters + 0.6, frontZ + 0.18)
    group.add(entranceCanopyShadow)

    const slimCanopy = new THREE.Mesh(
      new THREE.BoxGeometry(entryBayWidth + 0.18, 0.04, 0.84),
      darkMetalMat
    )
    slimCanopy.position.set(entryWorld.x, floorHeightMeters + 0.64, frontZ + 0.42)
    group.add(slimCanopy)
  }

  return group
}

function buildSingleFloorGroup({
  floor,
  floorIndex,
  totalFloors,
  offsetY,
  floorHeight,
  geometryMeta,
  materials,
  includeRoof,
  exteriorOnly = false,
}) {
  const floorGroup = new THREE.Group()
  floorGroup.name = `Floor_${floorIndex + 1}`

  const polygon = floor?.polygon || []
  if (polygon.length < 3) return floorGroup

  const transformedPolygon = polygon.map(([x, y]) => geometryMeta.transformPoint(x, y))
  const shape = polygonToShape(transformedPolygon)

  const slabGeo = new THREE.ExtrudeGeometry(shape, {
    depth: SLAB_THICKNESS,
    bevelEnabled: false,
    curveSegments: 1,
  })
  slabGeo.rotateX(-Math.PI / 2)

  const slab = new THREE.Mesh(slabGeo, materials.slabMaterial)
  slab.position.y = offsetY
  slab.receiveShadow = true
  slab.castShadow = true
  floorGroup.add(slab)

  if (!exteriorOnly) {
    const floorFinish = createInteriorFloorFinish({
      shape,
      floorIndex,
      yBase: offsetY + SLAB_THICKNESS,
    })
    floorGroup.add(floorFinish)
  }

  const wallGraph = buildWallGraph({
    polygon,
    innerWalls: floor.inner_walls || [],
  })

  const classifiedOpenings = classifyOpenings({
    graph: wallGraph,
    doors: floor.doors || [],
    windows: floor.windows || [],
    doorWidthPx: DOOR_OPENING_WIDTH_PX,
    windowWidthPx: WINDOW_OPENING_WIDTH_PX,
  })

  const { matched } = matchOpeningsToWalls({
    graph: wallGraph,
    openings: classifiedOpenings,
  })

  const openingsByWallId = new Map()
  matched.forEach((opening) => {
    if (!openingsByWallId.has(opening.wallId)) openingsByWallId.set(opening.wallId, [])
    openingsByWallId.get(opening.wallId).push(opening)
  })

  const { items: wallItems } = splitWallsByOpenings({
    graph: wallGraph,
    matchedOpenings: matched,
  })

  if (!exteriorOnly) {
    wallGraph.walls.forEach((graphWall) => {
      const isOuter = graphWall.kind === "outer"
      const wallOpenings = openingsByWallId.get(graphWall.id) || []
      const wallMesh = createWallSegmentMesh({
        line: graphWall.line,
        geometryMeta,
        height: isOuter ? floorHeight : floorHeight - 0.02,
        thickness: isOuter ? OUTER_WALL_THICKNESS : INNER_WALL_THICKNESS,
        material: isOuter ? materials.outerWallMaterial : materials.innerWallMaterial,
        yBase: offsetY + SLAB_THICKNESS,
        openings: wallOpenings,
      })
      if (wallMesh) floorGroup.add(wallMesh)
    })

    matched.forEach((opening) => {
      if (opening.type === "door") {
        const doorInsert = createDoorInsert({
          point: opening.point,
          line: opening.hostLine,
          geometryMeta,
          wallKind: opening.wallKind,
          widthPx: opening.widthPx,
          height: (opening.wallKind === "outer" ? floorHeight : floorHeight - 0.02) * DOOR_HEIGHT_RATIO,
          thickness: opening.wallKind === "outer" ? OUTER_WALL_THICKNESS : INNER_WALL_THICKNESS,
          yBase: offsetY + SLAB_THICKNESS,
          material: materials.doorMaterial,
          frameMaterial: materials.frameMaterial,
        })
        if (doorInsert) floorGroup.add(doorInsert)
        return
      }

      if (opening.type === "window") {
        const windowInsert = createWindowInsert({
          point: opening.point,
          line: opening.hostLine,
          geometryMeta,
          widthPx: opening.widthPx,
          height: WINDOW_HEIGHT,
          sillHeight: WINDOW_SILL_HEIGHT,
          thickness: OUTER_WALL_THICKNESS,
          yBase: offsetY + SLAB_THICKNESS,
          frameMaterial: materials.frameMaterial,
          glassMaterial: materials.glassMaterial,
        })
        if (windowInsert) floorGroup.add(windowInsert)
      }
    })

    const outline = createOutlineFromShape(shape, materials.lineMaterial)
    outline.position.y = offsetY + SLAB_THICKNESS + 0.003
    floorGroup.add(outline)

    if (includeRoof) {
      const roofGeo = new THREE.ExtrudeGeometry(shape, {
        depth: ROOF_THICKNESS,
        bevelEnabled: false,
        curveSegments: 1,
      })
      roofGeo.rotateX(-Math.PI / 2)

      const roof = new THREE.Mesh(roofGeo, materials.roofMaterial)
      roof.position.y = offsetY + SLAB_THICKNESS + floorHeight
      roof.receiveShadow = true
      roof.castShadow = true
      floorGroup.add(roof)
    }

    floorGroup.userData = {
      floorIndex,
      totalFloors,
    }

    return floorGroup
  }

  wallItems.forEach((item) => {
    if (item.type === "wall") {
      const isOuter = item.wallKind === "outer"
      if (exteriorOnly && !isOuter) return
      const wallMesh = createWallSegmentMesh({
        line: item.line,
        geometryMeta,
        height: isOuter ? floorHeight : floorHeight - 0.02,
        thickness: isOuter ? OUTER_WALL_THICKNESS : INNER_WALL_THICKNESS,
        material: isOuter ? materials.outerWallMaterial : materials.innerWallMaterial,
        yBase: offsetY + SLAB_THICKNESS,
      })
      if (wallMesh) floorGroup.add(wallMesh)
      return
    }

    if (item.type === "door") {
      const isOuter = item.wallKind === "outer"
      if (exteriorOnly && !isOuter) return
      const door = createDoorMesh({
        point: item.point,
        line: item.hostLine,
        geometryMeta,
        widthPx: item.widthPx,
        height: floorHeight * DOOR_HEIGHT_RATIO,
        thickness: isOuter ? OUTER_WALL_THICKNESS * 0.5 : INNER_WALL_THICKNESS * 0.7,
        yBase: offsetY + SLAB_THICKNESS,
        material: materials.doorMaterial,
      })
      if (door) floorGroup.add(door)
      return
    }

    if (item.type === "window") {
      if (exteriorOnly && item.wallKind !== "outer") return
      const windowGroup = createWindowGroup({
        point: item.point,
        line: item.hostLine,
        geometryMeta,
        widthPx: item.widthPx,
        height: WINDOW_HEIGHT,
        sillHeight: WINDOW_SILL_HEIGHT,
        thickness: OUTER_WALL_THICKNESS * 0.45,
        yBase: offsetY + SLAB_THICKNESS,
        frameMaterial: materials.frameMaterial,
        glassMaterial: materials.glassMaterial,
      })
      if (windowGroup) floorGroup.add(windowGroup)
    }
  })

  const outline = createOutlineFromShape(shape, materials.lineMaterial)
  if (!exteriorOnly) {
    outline.position.y = offsetY + SLAB_THICKNESS + 0.003
    floorGroup.add(outline)
  }

  if (includeRoof) {
    const roofGeo = new THREE.ExtrudeGeometry(shape, {
      depth: ROOF_THICKNESS,
      bevelEnabled: false,
      curveSegments: 1,
    })
    roofGeo.rotateX(-Math.PI / 2)

    const roof = new THREE.Mesh(roofGeo, materials.roofMaterial)
    roof.position.y = offsetY + SLAB_THICKNESS + floorHeight
    roof.receiveShadow = true
    roof.castShadow = true
    floorGroup.add(roof)
  }

  floorGroup.userData = {
    floorIndex,
    totalFloors,
  }

  return floorGroup
}

function createPitchedRoof({ polygon, geometryMeta, yBase, material }) {
  if (!polygon?.length) return null

  const pts = polygon.map(([x, y]) => geometryMeta.transformPoint(x, y))
  const xs = pts.map((p) => p.x)
  const zs = pts.map((p) => p.z)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minZ = Math.min(...zs)
  const maxZ = Math.max(...zs)
  const roofGroup = new THREE.Group()
  const slab = new THREE.Mesh(
    new THREE.BoxGeometry((maxX - minX) + 0.44, 0.08, (maxZ - minZ) + 0.44),
    material
  )
  slab.position.set((minX + maxX) / 2, yBase + 0.04, (minZ + maxZ) / 2)
  slab.castShadow = true
  slab.receiveShadow = true
  roofGroup.add(slab)

  const parapetMat = new THREE.MeshStandardMaterial({
    color: "#2d3338",
    roughness: 0.56,
    metalness: 0.26,
  })
  const parapetHeight = 0.12
  const parapetThickness = 0.05
  const width = maxX - minX
  const depth = maxZ - minZ

  const north = new THREE.Mesh(
    new THREE.BoxGeometry(width + 0.22, parapetHeight, parapetThickness),
    parapetMat
  )
  north.position.set((minX + maxX) / 2, yBase + parapetHeight / 2 + 0.1, minZ - 0.18)
  roofGroup.add(north)

  const south = new THREE.Mesh(
    new THREE.BoxGeometry(width + 0.22, parapetHeight, parapetThickness),
    parapetMat
  )
  south.position.set((minX + maxX) / 2, yBase + parapetHeight / 2 + 0.1, maxZ + 0.18)
  roofGroup.add(south)

  const west = new THREE.Mesh(
    new THREE.BoxGeometry(parapetThickness, parapetHeight, depth + 0.22),
    parapetMat
  )
  west.position.set(minX - 0.18, yBase + parapetHeight / 2 + 0.1, (minZ + maxZ) / 2)
  roofGroup.add(west)

  const east = new THREE.Mesh(
    new THREE.BoxGeometry(parapetThickness, parapetHeight, depth + 0.22),
    parapetMat
  )
  east.position.set(maxX + 0.18, yBase + parapetHeight / 2 + 0.1, (minZ + maxZ) / 2)
  roofGroup.add(east)

  const roofInset = new THREE.Mesh(
    new THREE.BoxGeometry(Math.max(1.6, width - 0.7), 0.02, Math.max(1.6, depth - 0.7)),
    new THREE.MeshStandardMaterial({
      color: "#b7aea0",
      roughness: 0.9,
      metalness: 0.0,
    })
  )
  roofInset.position.set((minX + maxX) / 2, yBase + 0.05, (minZ + maxZ) / 2)
  roofGroup.add(roofInset)

  const roofVolume = new THREE.Mesh(
    new THREE.BoxGeometry(Math.max(1.2, width * 0.14), 0.34, Math.max(1.4, depth * 0.11)),
    parapetMat
  )
  roofVolume.position.set((minX + maxX) / 2, yBase + 0.26, minZ + depth * 0.16)
  roofVolume.castShadow = true
  roofGroup.add(roofVolume)

  const chimney = new THREE.Mesh(
    new THREE.BoxGeometry(0.34, 0.72, 0.34),
    parapetMat
  )
  chimney.position.set(maxX - width * 0.14, yBase + 0.38, minZ + depth * 0.16)
  chimney.castShadow = true
  roofGroup.add(chimney)

  return roofGroup
}

function createEntranceFeature({ floor, geometryMeta, yBase, material }) {
  if (!floor?.polygon?.length || !geometryMeta) return null

  const polygon = floor.polygon
  const bottomY = Math.max(...polygon.map((pt) => pt[1]))
  const entrance = (floor.doors || [])
    .filter((door) => Math.abs(door.y - bottomY) < 28)
    .sort((a, b) => (b.width || 0) - (a.width || 0))[0]
  const fallback = {
    x: polygon.reduce((sum, pt) => sum + pt[0], 0) / polygon.length,
    y: bottomY,
    width: 108,
  }
  const entryPoint = entrance || fallback

  const center = geometryMeta.transformPoint(entryPoint.x, entryPoint.y)
  const group = new THREE.Group()
  const concreteSet = getConcreteTextureSet()
  const stoneSet = getStoneTextureSet()
  const metalSet = getDarkMetalTextureSet()
  const woodSet = getWoodAccentTextureSet()
  const stairMat = new THREE.MeshStandardMaterial({
    map: concreteSet.map.clone(),
    normalMap: concreteSet.normalMap.clone(),
    roughnessMap: concreteSet.roughnessMap.clone(),
    color: "#cdc6bb",
    roughness: 0.9,
    metalness: 0.0,
  })

  for (let i = 0; i < 3; i += 1) {
    const step = new THREE.Mesh(
      new THREE.BoxGeometry(2.24 - i * 0.18, 0.08, 0.34),
      stairMat
    )
    step.position.set(center.x, yBase + 0.04 + i * 0.08, center.z + 0.28 + i * 0.15)
    step.receiveShadow = true
    group.add(step)
  }

  const landing = new THREE.Mesh(
    new THREE.BoxGeometry(2.5, 0.06, 0.94),
    stairMat
  )
  landing.position.set(center.x, yBase + 0.24, center.z + 0.08)
  landing.receiveShadow = true
  group.add(landing)

  const portalMat = new THREE.MeshStandardMaterial({
    map: concreteSet.map.clone(),
    normalMap: concreteSet.normalMap.clone(),
    roughnessMap: concreteSet.roughnessMap.clone(),
    color: "#d5c7b3",
    roughness: 0.76,
    metalness: 0.02,
  })
  const stonePortalMat = new THREE.MeshStandardMaterial({
    map: stoneSet.map.clone(),
    normalMap: stoneSet.normalMap.clone(),
    roughnessMap: stoneSet.roughnessMap.clone(),
    color: "#736b63",
    roughness: 0.94,
    metalness: 0.0,
  })
  const portal = new THREE.Mesh(
    new THREE.BoxGeometry(2.92, 3.04, 0.18),
    portalMat
  )
  portal.position.set(center.x, yBase + 1.52, center.z + 0.14)
  portal.castShadow = true
  group.add(portal)

  const reveal = new THREE.Mesh(
    new THREE.BoxGeometry(2.18, 2.48, 0.08),
    new THREE.MeshStandardMaterial({
      map: metalSet.map.clone(),
      normalMap: metalSet.normalMap.clone(),
      roughnessMap: metalSet.roughnessMap.clone(),
      color: "#2c3239",
      roughness: 0.42,
      metalness: 0.35,
    })
  )
  reveal.position.set(center.x, yBase + 1.24, center.z + 0.23)
  group.add(reveal)

  const warmWall = new THREE.Mesh(
    new THREE.BoxGeometry(1.94, 2.22, 0.05),
    new THREE.MeshStandardMaterial({
      map: concreteSet.map.clone(),
      normalMap: concreteSet.normalMap.clone(),
      roughnessMap: concreteSet.roughnessMap.clone(),
      color: "#f1ebe2",
      roughness: 0.72,
      metalness: 0.0,
    })
  )
  warmWall.position.set(center.x, yBase + 1.1, center.z + 0.27)
  group.add(warmWall)

  const stonePortalSide = new THREE.Mesh(
    new THREE.BoxGeometry(0.34, 2.7, 0.1),
    stonePortalMat
  )
  stonePortalSide.position.set(center.x - 1.14, yBase + 1.34, center.z + 0.22)
  group.add(stonePortalSide)

  const mainDoorWidth = Math.max(1.08, (entryPoint.width || 108) * geometryMeta.scale * 0.82)
  const mainDoorHeight = 2.24
  const doorLeaf = new THREE.Mesh(
    new THREE.BoxGeometry(mainDoorWidth, mainDoorHeight, 0.12),
    new THREE.MeshStandardMaterial({
      map: woodSet.map.clone(),
      normalMap: woodSet.normalMap.clone(),
      roughnessMap: woodSet.roughnessMap.clone(),
      color: "#4c3528",
      roughness: 0.28,
      metalness: 0.16,
    })
  )
  doorLeaf.position.set(center.x, yBase + mainDoorHeight / 2, center.z + 0.16)
  doorLeaf.castShadow = true
  group.add(doorLeaf)

  const transom = new THREE.Mesh(
    new THREE.BoxGeometry(mainDoorWidth + 0.1, 0.22, 0.06),
    new THREE.MeshPhysicalMaterial({
      color: "#d8e8f0",
      roughness: 0.08,
      metalness: 0.0,
      transparent: true,
      opacity: 0.38,
      transmission: 0.9,
      thickness: 0.12,
      ior: 1.45,
    })
  )
  transom.position.set(center.x, yBase + mainDoorHeight + 0.18, center.z + 0.17)
  group.add(transom)

  const doorFrameMat = new THREE.MeshStandardMaterial({
    map: metalSet.map.clone(),
    normalMap: metalSet.normalMap.clone(),
    roughnessMap: metalSet.roughnessMap.clone(),
    color: "#171b20",
    roughness: 0.34,
    metalness: 0.48,
  })
  const frameLeft = new THREE.Mesh(
    new THREE.BoxGeometry(0.06, mainDoorHeight + 0.08, 0.1),
    doorFrameMat
  )
  frameLeft.position.set(center.x - mainDoorWidth / 2 - 0.05, yBase + mainDoorHeight / 2, center.z + 0.15)
  group.add(frameLeft)

  const frameRight = new THREE.Mesh(
    new THREE.BoxGeometry(0.06, mainDoorHeight + 0.08, 0.1),
    doorFrameMat
  )
  frameRight.position.set(center.x + mainDoorWidth / 2 + 0.05, yBase + mainDoorHeight / 2, center.z + 0.15)
  group.add(frameRight)

  const frameTop = new THREE.Mesh(
    new THREE.BoxGeometry(mainDoorWidth + 0.14, 0.08, 0.1),
    doorFrameMat
  )
  frameTop.position.set(center.x, yBase + mainDoorHeight + 0.05, center.z + 0.15)
  group.add(frameTop)

  const doorHandle = new THREE.Mesh(
    new THREE.BoxGeometry(0.04, 0.24, 0.04),
    new THREE.MeshStandardMaterial({
      color: "#c5ab7d",
      roughness: 0.28,
      metalness: 0.78,
    })
  )
  doorHandle.position.set(center.x + mainDoorWidth * 0.24, yBase + 1.05, center.z + 0.21)
  group.add(doorHandle)

  const canopy = new THREE.Mesh(
    new THREE.BoxGeometry(2.82, 0.05, 0.92),
    material
  )
  canopy.position.set(center.x, yBase + 2.72, center.z + 0.44)
  canopy.castShadow = true
  group.add(canopy)

  const canopyUnderside = new THREE.Mesh(
    new THREE.BoxGeometry(2.62, 0.03, 0.74),
    new THREE.MeshStandardMaterial({
      color: "#f5f1e9",
      roughness: 0.7,
      metalness: 0.02,
    })
  )
  canopyUnderside.position.set(center.x, yBase + 2.68, center.z + 0.38)
  group.add(canopyUnderside)

  const canopyTrim = new THREE.Mesh(
    new THREE.BoxGeometry(2.94, 0.04, 0.12),
    new THREE.MeshStandardMaterial({
      color: "#14191e",
      roughness: 0.3,
      metalness: 0.55,
    })
  )
  canopyTrim.position.set(center.x, yBase + 2.7, center.z + 0.88)
  group.add(canopyTrim)

  ;[-1, 1].forEach((side) => {
    const fin = new THREE.Mesh(
      new THREE.BoxGeometry(0.08, 2.42, 0.16),
      new THREE.MeshStandardMaterial({
        color: "#171b20",
        roughness: 0.32,
        metalness: 0.5,
      })
    )
    fin.position.set(center.x + side * 1.44, yBase + 1.36, center.z + 0.42)
    group.add(fin)

    const sconce = new THREE.Mesh(
      new THREE.BoxGeometry(0.06, 0.22, 0.05),
      new THREE.MeshStandardMaterial({
        color: "#f1eadf",
        emissive: "#c99d62",
        emissiveIntensity: 0.18,
        roughness: 0.6,
        metalness: 0.0,
      })
    )
    sconce.position.set(center.x + side * 0.92, yBase + 1.56, center.z + 0.33)
    group.add(sconce)
  })

  const threshold = new THREE.Mesh(
    new THREE.BoxGeometry(mainDoorWidth + 0.24, 0.03, 0.18),
    new THREE.MeshStandardMaterial({
      color: "#cabfae",
      roughness: 0.76,
      metalness: 0.0,
    })
  )
  threshold.position.set(center.x, yBase + 0.015, center.z + 0.22)
  group.add(threshold)

  return group
}

function polygonToShape(points) {
  const shape = new THREE.Shape()

  if (!points.length) return shape

  shape.moveTo(points[0].x, points[0].z)
  for (let i = 1; i < points.length; i += 1) {
    shape.lineTo(points[i].x, points[i].z)
  }
  shape.closePath()
  return shape
}

function createOutlineFromShape(shape, material) {
  const points = shape.getPoints()
  const geometry = new THREE.BufferGeometry().setFromPoints(
    points.map((p) => new THREE.Vector3(p.x, 0, p.y))
  )
  return new THREE.LineLoop(geometry, material)
}

function createWallSegmentMesh({
  line,
  geometryMeta,
  height,
  thickness,
  material,
  yBase,
  openings = [],
  castShadow = true,
  receiveShadow = true,
}) {
  const [x1, y1, x2, y2] = line
  const p1 = geometryMeta.transformPoint(x1, y1)
  const p2 = geometryMeta.transformPoint(x2, y2)

  const dx = p2.x - p1.x
  const dz = p2.z - p1.z
  const length = Math.hypot(dx, dz)

  if (length < 0.08) return null

  const group = new THREE.Group()
  const openingShapes = (openings || [])
    .map((opening) => buildWallOpeningRect({
      opening,
      length,
      height,
      scale: geometryMeta.scale,
    }))
    .filter(Boolean)

  const shape = new THREE.Shape()
  shape.moveTo(0, 0)
  shape.lineTo(length, 0)
  shape.lineTo(length, height)
  shape.lineTo(0, height)
  shape.closePath()

  openingShapes.forEach((rect) => {
    const hole = new THREE.Path()
    hole.moveTo(rect.x0, rect.y0)
    hole.lineTo(rect.x1, rect.y0)
    hole.lineTo(rect.x1, rect.y1)
    hole.lineTo(rect.x0, rect.y1)
    hole.closePath()
    shape.holes.push(hole)
  })

  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: thickness,
    bevelEnabled: false,
    curveSegments: 1,
    steps: 1,
  })
  geo.translate(-length / 2, 0, -thickness / 2)

  const mesh = new THREE.Mesh(geo, material)
  mesh.position.set((p1.x + p2.x) / 2, yBase, (p1.z + p2.z) / 2)
  mesh.rotation.y = Math.atan2(dz, dx)
  mesh.castShadow = castShadow
  mesh.receiveShadow = receiveShadow
  group.add(mesh)

  if (thickness >= OUTER_WALL_THICKNESS * 0.9) {
    const capMaterial = new THREE.MeshStandardMaterial({
      color: "#e7dfd4",
      roughness: 0.88,
      metalness: 0.0,
    })
    const topCap = new THREE.Mesh(
      new THREE.BoxGeometry(length, 0.06, thickness + 0.02),
      capMaterial
    )
    topCap.position.set((p1.x + p2.x) / 2, yBase + height - 0.03, (p1.z + p2.z) / 2)
    topCap.rotation.y = Math.atan2(dz, dx)
    group.add(topCap)
  } else {
    const trimMaterial = new THREE.MeshStandardMaterial({
      color: "#d4ccc0",
      roughness: 0.92,
      metalness: 0.0,
    })
    const topTrim = new THREE.Mesh(
      new THREE.BoxGeometry(length, 0.035, thickness + 0.01),
      trimMaterial
    )
    topTrim.position.set((p1.x + p2.x) / 2, yBase + height - 0.02, (p1.z + p2.z) / 2)
    topTrim.rotation.y = Math.atan2(dz, dx)
    group.add(topTrim)
  }

  return group
}

function buildWallOpeningRect({ opening, length, height, scale }) {
  const center = length * opening.t
  const width = getOpeningWidthWorld(opening, scale)

  const x0 = Math.max(0.08, center - width / 2)
  const x1 = Math.min(length - 0.08, center + width / 2)

  if (x1 - x0 < 0.18) return null

  if (opening.type === "door") {
    const y1 = Math.min(height - 0.14, 2.08)
    return {
      x0,
      x1,
      y0: 0,
      y1,
    }
  }

  const y0 = WINDOW_SILL_HEIGHT
  const y1 = Math.min(height - 0.22, 1.96)

  if (y1 - y0 < 0.18) return null

  return {
    x0,
    x1,
    y0,
    y1,
  }
}

function getOpeningWidthWorld(opening, scale) {
  return opening.type === "door"
    ? Math.max(0.46, opening.widthPx * scale * 0.36)
    : Math.max(0.68, opening.widthPx * scale * 0.42)
}

function validateOpeningsAgainstRooms({ floor, openings }) {
  const rooms = floor?.rooms || []
  if (!rooms.length) return openings

  return openings.filter((opening) => {
    if (opening.type === "window") {
      return opening.wallKind === "outer"
    }

    if (opening.wallKind === "outer") {
      return true
    }

    const adjacentRooms = findRoomsAdjacentToOpening(rooms, opening)
    return adjacentRooms.length >= 2
  })
}

function findRoomsAdjacentToOpening(rooms, opening) {
  const [x1, y1, x2, y2] = opening.hostLine
  const horizontal = Math.abs(y1 - y2) <= Math.abs(x1 - x2)
  const halfSpan = Math.max(26, (opening.widthPx || 80) * 0.7)
  const band = 110

  const roomSides = rooms
    .map((room) => {
      const along = horizontal ? room.x - opening.point.x : room.y - opening.point.y
      const across = horizontal ? room.y - opening.point.y : room.x - opening.point.x

      if (Math.abs(along) > halfSpan || Math.abs(across) > band) {
        return null
      }

      return {
        room,
        side: across < 0 ? "negative" : "positive",
        score: Math.abs(along) + Math.abs(across) * 0.35,
      }
    })
    .filter(Boolean)
    .sort((a, b) => a.score - b.score)

  const negative = roomSides.find((item) => item.side === "negative")
  const positive = roomSides.find((item) => item.side === "positive")

  const result = []
  if (negative) result.push(negative.room)
  if (positive) result.push(positive.room)
  return result
}

function createDoorMesh({
  point,
  line,
  geometryMeta,
  widthPx,
  height,
  thickness,
  yBase,
  material,
}) {
  if (!geometryMeta) return null

  const [x1, y1, x2, y2] = line
  const p1 = geometryMeta.transformPoint(x1, y1)
  const p2 = geometryMeta.transformPoint(x2, y2)
  const center = geometryMeta.transformPoint(point.x, point.y)

  const dir = new THREE.Vector3(p2.x - p1.x, 0, p2.z - p1.z)
  const len = dir.length()
  if (len < 0.001) return null

  dir.normalize()

  const scaledWidth = Math.max(0.44, widthPx * geometryMeta.scale * 0.36)
  const group = new THREE.Group()
  const insetDepth = Math.max(0.03, thickness * 0.2)
  const topFillHeight = Math.max(0.3, 2.38 - height)

  const leaf = new THREE.Mesh(
    new THREE.BoxGeometry(scaledWidth, height, Math.max(0.05, thickness * 0.44)),
    material
  )
  leaf.position.set(0, height / 2, -insetDepth)
  leaf.castShadow = true
  leaf.receiveShadow = true
  group.add(leaf)

  const frameMaterial = new THREE.MeshStandardMaterial({
    color: "#e9e1d6",
    roughness: 0.82,
    metalness: 0.02,
  })

  const leftJamb = new THREE.Mesh(
    new THREE.BoxGeometry(0.06, height + 0.02, thickness + 0.04),
    frameMaterial
  )
  leftJamb.position.set(-scaledWidth / 2 - 0.035, height / 2, 0)
  group.add(leftJamb)

  const rightJamb = new THREE.Mesh(
    new THREE.BoxGeometry(0.06, height + 0.02, thickness + 0.04),
    frameMaterial
  )
  rightJamb.position.set(scaledWidth / 2 + 0.035, height / 2, 0)
  group.add(rightJamb)

  const lintel = new THREE.Mesh(
    new THREE.BoxGeometry(scaledWidth + 0.12, 0.1, thickness + 0.04),
    frameMaterial
  )
  lintel.position.set(0, height + 0.04, 0)
  group.add(lintel)

  const revealMaterial = new THREE.MeshStandardMaterial({
    color: "#d9d1c5",
    roughness: 0.88,
    metalness: 0.0,
  })

  const leftReveal = new THREE.Mesh(
    new THREE.BoxGeometry(0.09, height, Math.max(0.08, thickness * 0.82)),
    revealMaterial
  )
  leftReveal.position.set(-scaledWidth / 2 - 0.08, height / 2, -insetDepth * 0.4)
  group.add(leftReveal)

  const rightReveal = new THREE.Mesh(
    new THREE.BoxGeometry(0.09, height, Math.max(0.08, thickness * 0.82)),
    revealMaterial
  )
  rightReveal.position.set(scaledWidth / 2 + 0.08, height / 2, -insetDepth * 0.4)
  group.add(rightReveal)

  const headWall = new THREE.Mesh(
    new THREE.BoxGeometry(scaledWidth + 0.18, topFillHeight, Math.max(0.08, thickness * 0.88)),
    revealMaterial
  )
  headWall.position.set(0, height + topFillHeight / 2, -insetDepth * 0.25)
  group.add(headWall)

  const threshold = new THREE.Mesh(
    new THREE.BoxGeometry(scaledWidth + 0.08, 0.03, Math.max(0.08, thickness * 0.56)),
    frameMaterial
  )
  threshold.position.set(0, 0.015, -insetDepth * 0.18)
  group.add(threshold)

  const handle = new THREE.Mesh(
    new THREE.BoxGeometry(0.04, 0.16, 0.03),
    new THREE.MeshStandardMaterial({
      color: "#c4ac7f",
      roughness: 0.35,
      metalness: 0.7,
    })
  )
  handle.position.set(scaledWidth * 0.28, height * 0.48, thickness * 0.18)
  group.add(handle)

  group.position.set(center.x, yBase, center.z)
  group.rotation.y = Math.atan2(dir.z, dir.x)
  return group
}

function createWindowGroup({
  point,
  line,
  geometryMeta,
  widthPx,
  height,
  sillHeight,
  thickness,
  yBase,
  frameMaterial,
  glassMaterial,
}) {
  if (!geometryMeta) return null

  const [x1, y1, x2, y2] = line
  const p1 = geometryMeta.transformPoint(x1, y1)
  const p2 = geometryMeta.transformPoint(x2, y2)
  const center = geometryMeta.transformPoint(point.x, point.y)

  const dir = new THREE.Vector3(p2.x - p1.x, 0, p2.z - p1.z)
  const len = dir.length()
  if (len < 0.001) return null

  dir.normalize()

  const group = new THREE.Group()
  const widthWorld = Math.max(0.56, widthPx * geometryMeta.scale * 0.38)
  const frameDepth = Math.max(0.06, thickness * 0.24)
  const frameSide = 0.05
  const frameTopBottom = 0.05
  const insetDepth = Math.max(0.02, thickness * 0.12)
  const openingDepth = Math.max(0.1, thickness * 0.82)
  const wallMaterial = new THREE.MeshStandardMaterial({
    color: "#d8d0c4",
    roughness: 0.9,
    metalness: 0.0,
  })
  const sidePier = 0.14
  const headHeight = Math.max(0.34, 2.42 - (sillHeight + height))

  const leftPier = new THREE.Mesh(
    new THREE.BoxGeometry(sidePier, height + 0.04, openingDepth),
    wallMaterial
  )
  leftPier.position.set(-widthWorld / 2 - sidePier / 2, sillHeight + height / 2, 0)
  group.add(leftPier)

  const rightPier = new THREE.Mesh(
    new THREE.BoxGeometry(sidePier, height + 0.04, openingDepth),
    wallMaterial
  )
  rightPier.position.set(widthWorld / 2 + sidePier / 2, sillHeight + height / 2, 0)
  group.add(rightPier)

  const bottomWall = new THREE.Mesh(
    new THREE.BoxGeometry(widthWorld + sidePier * 2, sillHeight, openingDepth),
    wallMaterial
  )
  bottomWall.position.set(0, sillHeight / 2, 0)
  group.add(bottomWall)

  const headWall = new THREE.Mesh(
    new THREE.BoxGeometry(widthWorld + sidePier * 2, headHeight, openingDepth),
    wallMaterial
  )
  headWall.position.set(0, sillHeight + height + headHeight / 2, 0)
  group.add(headWall)

  const glassGeo = new THREE.BoxGeometry(
    Math.max(0.1, widthWorld - frameSide * 2),
    Math.max(0.1, height - frameTopBottom * 2),
    Math.max(0.03, frameDepth)
  )
  const glass = new THREE.Mesh(glassGeo, glassMaterial)
  glass.position.set(0, sillHeight + height / 2, -insetDepth)
  glass.castShadow = false
  glass.receiveShadow = true
  group.add(glass)

  const interiorShadow = new THREE.Mesh(
    new THREE.BoxGeometry(
      Math.max(0.08, widthWorld - frameSide * 2 - 0.04),
      Math.max(0.08, height - frameTopBottom * 2 - 0.04),
      Math.max(0.02, frameDepth * 0.55)
    ),
    new THREE.MeshStandardMaterial({
      color: "#5d5a56",
      roughness: 0.96,
      metalness: 0.0,
    })
  )
  interiorShadow.position.set(0, sillHeight + height / 2, -insetDepth - 0.045)
  group.add(interiorShadow)

  const mullion = new THREE.Mesh(
    new THREE.BoxGeometry(0.035, Math.max(0.24, height - frameTopBottom * 2), frameDepth * 0.92),
    frameMaterial
  )
  mullion.position.set(0, sillHeight + height / 2, -insetDepth)
  group.add(mullion)

  const leftFrame = new THREE.Mesh(
    new THREE.BoxGeometry(frameSide, height, frameDepth),
    frameMaterial
  )
  leftFrame.position.set(-widthWorld / 2 + frameSide / 2, sillHeight + height / 2, -insetDepth)
  group.add(leftFrame)

  const rightFrame = new THREE.Mesh(
    new THREE.BoxGeometry(frameSide, height, frameDepth),
    frameMaterial
  )
  rightFrame.position.set(widthWorld / 2 - frameSide / 2, sillHeight + height / 2, -insetDepth)
  group.add(rightFrame)

  const topFrame = new THREE.Mesh(
    new THREE.BoxGeometry(widthWorld, frameTopBottom, frameDepth),
    frameMaterial
  )
  topFrame.position.set(0, sillHeight + height - frameTopBottom / 2, -insetDepth)
  group.add(topFrame)

  const bottomFrame = new THREE.Mesh(
    new THREE.BoxGeometry(widthWorld, frameTopBottom, frameDepth),
    frameMaterial
  )
  bottomFrame.position.set(0, sillHeight + frameTopBottom / 2, -insetDepth)
  group.add(bottomFrame)

  const sill = new THREE.Mesh(
    new THREE.BoxGeometry(widthWorld + 0.18, 0.055, frameDepth + 0.12),
    frameMaterial
  )
  sill.position.set(0, sillHeight - 0.025, thickness * 0.26)
  group.add(sill)

  const sillShadow = new THREE.Mesh(
    new THREE.BoxGeometry(widthWorld + 0.1, 0.025, frameDepth + 0.04),
    new THREE.MeshStandardMaterial({
      color: "#b7afa4",
      roughness: 0.96,
      metalness: 0.0,
    })
  )
  sillShadow.position.set(0, sillHeight - 0.055, thickness * 0.3)
  group.add(sillShadow)

  group.position.set(center.x, yBase, center.z)
  group.rotation.y = Math.atan2(dir.z, dir.x)

  return group
}

function createDoorInsert({
  point,
  line,
  geometryMeta,
  wallKind,
  widthPx,
  height,
  thickness,
  yBase,
  material,
  frameMaterial,
}) {
  if (!geometryMeta) return null

  const [x1, y1, x2, y2] = line
  const p1 = geometryMeta.transformPoint(x1, y1)
  const p2 = geometryMeta.transformPoint(x2, y2)
  const center = geometryMeta.transformPoint(point.x, point.y)

  const dir = new THREE.Vector3(p2.x - p1.x, 0, p2.z - p1.z)
  if (dir.length() < 0.001) return null
  dir.normalize()

  const widthWorld = getOpeningWidthWorld({ type: "door", widthPx }, geometryMeta.scale) - 0.04
  const panelDepth = Math.max(0.035, thickness * 0.14)
  const group = new THREE.Group()
  const leafHeight = Math.min(height - 0.04, 2.04)
  const isInnerDoor = wallKind === "inner"

  const leaf = new THREE.Mesh(
    new THREE.BoxGeometry(widthWorld, leafHeight, panelDepth),
    material
  )
  leaf.castShadow = true
  leaf.receiveShadow = true

  if (isInnerDoor) {
    leaf.position.set(0, leafHeight / 2, -thickness * 0.08)
    group.add(leaf)
  } else {
    const hingeGroup = new THREE.Group()
    hingeGroup.position.set(-widthWorld / 2, 0, 0)
    hingeGroup.rotation.y = -0.32
    group.add(hingeGroup)
    leaf.position.set(widthWorld / 2, leafHeight / 2, -thickness * 0.08)
    hingeGroup.add(leaf)

    const handle = new THREE.Mesh(
      new THREE.BoxGeometry(0.03, 0.14, 0.02),
      new THREE.MeshStandardMaterial({
        color: "#c5ab7d",
        roughness: 0.3,
        metalness: 0.76,
      })
    )
    handle.position.set(widthWorld * 0.32, leafHeight * 0.5, panelDepth * 0.9)
    hingeGroup.add(handle)
  }

  const jambWidth = 0.045
  const jambDepth = Math.max(0.06, thickness * 0.42)
  const leftJamb = new THREE.Mesh(
    new THREE.BoxGeometry(jambWidth, leafHeight + 0.03, jambDepth),
    frameMaterial
  )
  leftJamb.position.set(-widthWorld / 2 - jambWidth / 2, leafHeight / 2, 0)
  group.add(leftJamb)

  const rightJamb = new THREE.Mesh(
    new THREE.BoxGeometry(jambWidth, leafHeight + 0.03, jambDepth),
    frameMaterial
  )
  rightJamb.position.set(widthWorld / 2 + jambWidth / 2, leafHeight / 2, 0)
  group.add(rightJamb)

  const lintel = new THREE.Mesh(
    new THREE.BoxGeometry(widthWorld + jambWidth * 2, 0.05, jambDepth),
    frameMaterial
  )
  lintel.position.set(0, leafHeight + 0.025, 0)
  group.add(lintel)

  if (isInnerDoor) {
    const handleLeft = new THREE.Mesh(
      new THREE.BoxGeometry(0.025, 0.12, 0.02),
      new THREE.MeshStandardMaterial({
        color: "#c5ab7d",
        roughness: 0.3,
        metalness: 0.76,
      })
    )
    handleLeft.position.set(widthWorld * 0.28, leafHeight * 0.48, panelDepth * 0.9)
    group.add(handleLeft)
  }

  group.position.set(center.x, yBase, center.z)
  group.rotation.y = Math.atan2(dir.z, dir.x)
  return group
}

function createWindowInsert({
  point,
  line,
  geometryMeta,
  widthPx,
  height,
  sillHeight,
  thickness,
  yBase,
  frameMaterial,
  glassMaterial,
}) {
  if (!geometryMeta) return null

  const [x1, y1, x2, y2] = line
  const p1 = geometryMeta.transformPoint(x1, y1)
  const p2 = geometryMeta.transformPoint(x2, y2)
  const center = geometryMeta.transformPoint(point.x, point.y)

  const dir = new THREE.Vector3(p2.x - p1.x, 0, p2.z - p1.z)
  if (dir.length() < 0.001) return null
  dir.normalize()

  const widthWorld = getOpeningWidthWorld({ type: "window", widthPx }, geometryMeta.scale) - 0.06
  const frameDepth = Math.max(0.045, thickness * 0.12)
  const paneDepth = Math.max(0.022, thickness * 0.05)
  const frameSide = 0.045
  const frameTopBottom = 0.045
  const group = new THREE.Group()

  const glass = new THREE.Mesh(
    new THREE.BoxGeometry(
      Math.max(0.1, widthWorld - frameSide * 2),
      Math.max(0.1, height - frameTopBottom * 2),
      paneDepth
    ),
    glassMaterial
  )
  glass.position.set(0, sillHeight + height / 2, -thickness * 0.08)
  glass.receiveShadow = true
  group.add(glass)

  const interiorShadow = new THREE.Mesh(
    new THREE.BoxGeometry(
      Math.max(0.08, widthWorld - frameSide * 2 - 0.04),
      Math.max(0.08, height - frameTopBottom * 2 - 0.04),
      Math.max(0.018, paneDepth * 0.9)
    ),
    new THREE.MeshStandardMaterial({
      color: "#605d59",
      roughness: 0.96,
      metalness: 0.0,
    })
  )
  interiorShadow.position.set(0, sillHeight + height / 2, -thickness * 0.12)
  group.add(interiorShadow)

  const mullion = new THREE.Mesh(
    new THREE.BoxGeometry(0.03, Math.max(0.2, height - frameTopBottom * 2), frameDepth),
    frameMaterial
  )
  mullion.position.set(0, sillHeight + height / 2, -thickness * 0.08)
  group.add(mullion)

  const leftFrame = new THREE.Mesh(
    new THREE.BoxGeometry(frameSide, height, frameDepth),
    frameMaterial
  )
  leftFrame.position.set(-widthWorld / 2 + frameSide / 2, sillHeight + height / 2, -thickness * 0.08)
  group.add(leftFrame)

  const rightFrame = new THREE.Mesh(
    new THREE.BoxGeometry(frameSide, height, frameDepth),
    frameMaterial
  )
  rightFrame.position.set(widthWorld / 2 - frameSide / 2, sillHeight + height / 2, -thickness * 0.08)
  group.add(rightFrame)

  const topFrame = new THREE.Mesh(
    new THREE.BoxGeometry(widthWorld, frameTopBottom, frameDepth),
    frameMaterial
  )
  topFrame.position.set(0, sillHeight + height - frameTopBottom / 2, -thickness * 0.08)
  group.add(topFrame)

  const bottomFrame = new THREE.Mesh(
    new THREE.BoxGeometry(widthWorld, frameTopBottom, frameDepth),
    frameMaterial
  )
  bottomFrame.position.set(0, sillHeight + frameTopBottom / 2, -thickness * 0.08)
  group.add(bottomFrame)

  group.position.set(center.x, yBase, center.z)
  group.rotation.y = Math.atan2(dir.z, dir.x)
  return group
}

const shellStyle = {
  minHeight: "100vh",
  background:
    "radial-gradient(circle at 14% 12%, rgba(137,92,246,0.22), transparent 26%), radial-gradient(circle at 86% 20%, rgba(34,211,238,0.16), transparent 24%), radial-gradient(circle at 52% 82%, rgba(249,115,22,0.08), transparent 24%), linear-gradient(180deg,#070b16 0%,#0c1323 34%,#111d31 68%,#0b1626 100%)",
  color: "#f6f7f8",
  padding: 26,
  fontFamily: '"Segoe UI", "Helvetica Neue", Arial, sans-serif',
  position: "relative",
  overflow: "hidden",
}

const pageStyle = {
  maxWidth: 1600,
  margin: "0 auto",
  position: "relative",
  zIndex: 1,
}

const topBarStyle = {
  position: "sticky",
  top: 16,
  zIndex: 6,
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  gap: 20,
  marginBottom: 22,
  padding: "16px 18px",
  borderRadius: 24,
  background: "rgba(10, 16, 28, 0.52)",
  border: "1px solid rgba(255,255,255,0.08)",
  backdropFilter: "blur(18px)",
  boxShadow: "0 22px 54px rgba(0,0,0,0.18)",
}

const topBarCompactStyle = {
  flexDirection: "column",
  alignItems: "flex-start",
}

const brandStyle = {
  fontSize: 30,
  fontWeight: 900,
  color: "#f7f8f9",
  letterSpacing: "-0.04em",
}

const brandSubStyle = {
  marginTop: 4,
  color: "#8ba2bf",
  fontSize: 12,
  letterSpacing: "0.14em",
  textTransform: "uppercase",
}

const profileCardStyle = {
  display: "flex",
  alignItems: "center",
  gap: 12,
  padding: "10px 12px",
  borderRadius: 16,
  background: "linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04))",
  border: "1px solid rgba(255,255,255,0.1)",
  boxShadow: "0 20px 40px rgba(0,0,0,0.18)",
  backdropFilter: "blur(16px)",
}

const profileAvatarStyle = {
  width: 42,
  height: 42,
  borderRadius: "50%",
  display: "grid",
  placeItems: "center",
  background: "linear-gradient(135deg,#5a829a,#d2b57e)",
  color: "#09111a",
  fontWeight: 800,
}

const profileLabelStyle = {
  color: "#8fa5b4",
  fontSize: 11,
  textTransform: "uppercase",
  letterSpacing: "0.12em",
}

const profileNameStyle = {
  color: "#ffffff",
  fontSize: 15,
  fontWeight: 700,
}

const backgroundGlowOne = {
  position: "fixed",
  width: 420,
  height: 420,
  borderRadius: "50%",
  top: -120,
  left: -100,
  background: "radial-gradient(circle, rgba(148,190,214,0.18), transparent 70%)",
  pointerEvents: "none",
  filter: "blur(16px)",
}

const backgroundGlowTwo = {
  position: "fixed",
  width: 620,
  height: 620,
  borderRadius: "50%",
  right: -180,
  bottom: -220,
  background: "radial-gradient(circle, rgba(56,189,248,0.12), transparent 72%)",
  pointerEvents: "none",
  filter: "blur(34px)",
}

const landingShellStyle = {
  ...shellStyle,
  padding: 0,
}

const landingPageStyle = {
  maxWidth: 1480,
  margin: "0 auto",
  padding: "26px 28px 56px",
  position: "relative",
  zIndex: 1,
}

const landingTopBarStyle = {
  position: "sticky",
  top: 16,
  zIndex: 4,
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: 18,
  padding: "16px 20px",
  marginBottom: 28,
  borderRadius: 24,
  background: "rgba(8, 14, 24, 0.52)",
  border: "1px solid rgba(255,255,255,0.08)",
  backdropFilter: "blur(18px)",
  boxShadow: "0 18px 44px rgba(0,0,0,0.2)",
}

const landingTopBarCompactStyle = {
  flexDirection: "column",
  alignItems: "flex-start",
}

const landingHeroGridStyle = {
  display: "grid",
  gridTemplateColumns: "1fr",
  gap: 28,
  alignItems: "center",
  marginBottom: 36,
}

const landingHeroGridCompactStyle = {
  gridTemplateColumns: "1fr",
}

const landingBadgeStyle = {
  display: "inline-flex",
  alignItems: "center",
  gap: 10,
  padding: "10px 16px",
  borderRadius: 999,
  background: "linear-gradient(135deg, rgba(122,92,255,0.22), rgba(34,211,238,0.14))",
  border: "1px solid rgba(255,255,255,0.12)",
  color: "#d8e9ff",
  fontSize: 12,
  fontWeight: 800,
  letterSpacing: "0.14em",
  textTransform: "uppercase",
  marginBottom: 18,
}

const landingTitleStyle = {
  margin: 0,
  maxWidth: 760,
  fontSize: "clamp(3rem, 5vw, 5.2rem)",
  lineHeight: 0.94,
  letterSpacing: "-0.05em",
  fontWeight: 900,
  color: "#fbfdff",
}

const landingTextStyle = {
  marginTop: 18,
  maxWidth: 680,
  color: "#b8c6d8",
  fontSize: 18,
  lineHeight: 1.8,
}

const landingActionRowStyle = {
  display: "flex",
  gap: 14,
  flexWrap: "wrap",
  marginTop: 24,
}

const featureStripStyle = {
  display: "flex",
  gap: 12,
  flexWrap: "wrap",
  marginTop: 26,
}

const featureStripCardStyle = {
  padding: "12px 16px",
  borderRadius: 18,
  background: "rgba(255,255,255,0.05)",
  border: "1px solid rgba(255,255,255,0.08)",
  backdropFilter: "blur(14px)",
  color: "#e7efff",
  fontSize: 13,
  fontWeight: 700,
  boxShadow: "0 12px 28px rgba(0,0,0,0.12)",
}

const landingPreviewCardStyle = {
  padding: 22,
  borderRadius: 30,
  background: "linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03))",
  border: "1px solid rgba(255,255,255,0.1)",
  backdropFilter: "blur(18px)",
  boxShadow: "0 30px 64px rgba(0,0,0,0.24)",
}

const landingPreviewBadgeStyle = {
  display: "inline-flex",
  padding: "8px 12px",
  borderRadius: 999,
  background: "rgba(255,255,255,0.06)",
  border: "1px solid rgba(255,255,255,0.08)",
  color: "#d2deef",
  fontSize: 12,
  fontWeight: 700,
  marginBottom: 16,
}

const landingPreviewFrameStyle = {
  display: "grid",
  gridTemplateColumns: "0.92fr 1.08fr",
  gap: 16,
  alignItems: "stretch",
}

const landingPlanPreviewStyle = {
  width: "100%",
  height: 420,
  objectFit: "contain",
  borderRadius: 22,
  background: "#ffffff",
  boxShadow: "inset 0 0 0 1px rgba(15,23,32,0.08)",
}

const landingPreviewRenderStyle = {
  width: "100%",
  height: 420,
  objectFit: "cover",
  borderRadius: 24,
  background: "#dce3ea",
  border: "1px solid rgba(255,255,255,0.16)",
}

const landingPreviewMockStyle = {
  borderRadius: 24,
  minHeight: 420,
  background:
    "radial-gradient(circle at top, rgba(176,231,255,0.55), rgba(215,224,236,0.82) 44%, rgba(221,216,209,1) 100%)",
  border: "1px solid rgba(255,255,255,0.16)",
  overflow: "hidden",
  padding: 22,
  display: "flex",
  flexDirection: "column",
}

const landingMockHeaderStyle = {
  color: "#3a4c60",
  fontSize: 12,
  fontWeight: 800,
  letterSpacing: "0.12em",
  textTransform: "uppercase",
  marginBottom: 20,
}

const landingMockStageStyle = {
  flex: 1,
  borderRadius: 24,
  background: "linear-gradient(180deg, rgba(244,244,242,0.74), rgba(213,205,195,0.96))",
  position: "relative",
  overflow: "hidden",
  display: "grid",
  placeItems: "center",
}

const landingMockBuildingStyle = {
  width: 240,
  height: 190,
  borderRadius: 18,
  background: "linear-gradient(180deg, #f5f2eb 0%, #d7cebf 100%)",
  boxShadow: "0 32px 58px rgba(84,76,64,0.2)",
  position: "relative",
  transform: "perspective(900px) rotateX(16deg) rotateZ(-10deg)",
}

const landingMockRoofStyle = {
  position: "absolute",
  inset: "-8px -8px auto",
  height: 18,
  borderRadius: "16px 16px 10px 10px",
  background: "#1a212e",
}

const landingMockWindowRowStyle = {
  position: "absolute",
  left: 28,
  right: 28,
  display: "flex",
  justifyContent: "space-between",
  gap: 16,
}

landingMockWindowRowStyle.top = 40

const landingMockDoorStyle = {
  position: "absolute",
  left: "50%",
  bottom: 24,
  width: 52,
  height: 76,
  transform: "translateX(-50%)",
  borderRadius: "10px 10px 4px 4px",
  background: "linear-gradient(180deg,#6c4a39,#4e3328)",
}

const landingSectionStyle = {
  marginTop: 22,
  padding: 24,
  borderRadius: 28,
  background: "rgba(255,255,255,0.04)",
  border: "1px solid rgba(255,255,255,0.07)",
  backdropFilter: "blur(16px)",
}

const landingSectionHeaderStyle = {
  fontSize: 28,
  fontWeight: 800,
  letterSpacing: "-0.03em",
  color: "#fbfdff",
  marginBottom: 18,
}

const landingCardGridStyle = {
  display: "grid",
  gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
  gap: 18,
}

const landingCardGridCompactStyle = {
  gridTemplateColumns: "1fr",
}

const landingInfoCardStyle = {
  padding: 22,
  borderRadius: 24,
  background: "linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03))",
  border: "1px solid rgba(255,255,255,0.08)",
  boxShadow: "0 20px 46px rgba(0,0,0,0.12)",
}

const landingStepPillStyle = {
  display: "inline-flex",
  padding: "7px 12px",
  borderRadius: 999,
  background: "linear-gradient(135deg, rgba(139,92,246,0.22), rgba(34,211,238,0.18))",
  border: "1px solid rgba(255,255,255,0.1)",
  color: "#deebff",
  fontSize: 12,
  fontWeight: 800,
  marginBottom: 16,
}

const landingInfoTitleStyle = {
  fontSize: 18,
  fontWeight: 800,
  color: "#f7fafe",
  marginBottom: 10,
}

const landingInfoTextStyle = {
  color: "#aebfd1",
  fontSize: 14,
  lineHeight: 1.75,
}

const landingSplitSectionStyle = {
  marginTop: 26,
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: 18,
}

const landingSplitSectionCompactStyle = {
  gridTemplateColumns: "1fr",
}

const landingSplitCardStyle = {
  padding: 22,
  borderRadius: 28,
  background: "rgba(255,255,255,0.04)",
  border: "1px solid rgba(255,255,255,0.08)",
  backdropFilter: "blur(16px)",
}

const landingBeforeAfterImageStyle = {
  width: "100%",
  height: 360,
  objectFit: "contain",
  background: "#ffffff",
  borderRadius: 22,
  marginTop: 14,
}

const landingAfterMockStyle = {
  height: 360,
  borderRadius: 24,
  marginTop: 14,
  background:
    "radial-gradient(circle at top, rgba(213,234,245,0.9), rgba(228,220,210,0.96) 60%, rgba(221,214,204,1) 100%)",
  display: "grid",
  placeItems: "center",
  overflow: "hidden",
}

const landingAfterBuildingStyle = {
  width: 260,
  height: 180,
  borderRadius: 20,
  background: "linear-gradient(180deg, #f3eee6 0%, #d8cfbf 100%)",
  position: "relative",
  boxShadow: "0 28px 54px rgba(77,67,52,0.18)",
  transform: "perspective(900px) rotateX(18deg) rotateZ(-8deg)",
}

const landingCtaStyle = {
  marginTop: 28,
  padding: 28,
  borderRadius: 30,
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  gap: 18,
  background:
    "linear-gradient(135deg, rgba(122,92,255,0.18), rgba(34,211,238,0.12), rgba(255,255,255,0.04))",
  border: "1px solid rgba(255,255,255,0.08)",
  boxShadow: "0 24px 56px rgba(0,0,0,0.14)",
}

const topBarRightStyle = {
  display: "flex",
  alignItems: "center",
  gap: 14,
  flexWrap: "wrap",
}

const topBarRightCompactStyle = {
  width: "100%",
  justifyContent: "space-between",
}

const headerActionPillStyle = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  padding: "8px",
  borderRadius: 18,
  background: "rgba(255,255,255,0.05)",
  border: "1px solid rgba(255,255,255,0.08)",
  backdropFilter: "blur(14px)",
}

const headerIconButtonStyle = {
  padding: "10px 14px",
  borderRadius: 12,
  border: "1px solid rgba(255,255,255,0.08)",
  background: "linear-gradient(135deg, rgba(139,92,246,0.18), rgba(34,211,238,0.12))",
  color: "#ecf5ff",
  fontWeight: 700,
  cursor: "pointer",
}

const heroStyle = {
  marginBottom: 28,
}

const heroHeaderRow = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "flex-start",
  gap: 24,
  flexWrap: "wrap",
}

const badgeStyle = {
  display: "inline-flex",
  alignItems: "center",
  gap: 8,
  padding: "8px 14px",
  borderRadius: 999,
  border: "1px solid rgba(163,194,214,0.22)",
  background: "rgba(255,255,255,0.05)",
  color: "#b8d7e9",
  fontSize: 12,
  fontWeight: 700,
  letterSpacing: "0.12em",
  textTransform: "uppercase",
  marginBottom: 16,
  backdropFilter: "blur(14px)",
}

const heroTitleStyle = {
  margin: 0,
  fontSize: "clamp(2.2rem, 3vw, 4rem)",
  lineHeight: 1.02,
  fontWeight: 800,
  color: "#f6f7f8",
  maxWidth: 860,
}

const heroTextStyle = {
  marginTop: 14,
  maxWidth: 820,
  color: "#b8c5cf",
  fontSize: 17,
  lineHeight: 1.8,
}

const heroChipRowStyle = {
  marginTop: 18,
  display: "flex",
  gap: 10,
  flexWrap: "wrap",
}

const heroChipStyle = {
  padding: "9px 14px",
  borderRadius: 999,
  background: "rgba(255,255,255,0.05)",
  border: "1px solid rgba(255,255,255,0.08)",
  color: "#d3e0e8",
  fontSize: 13,
  fontWeight: 600,
}

const loadingCardStyle = {
  marginTop: 18,
  width: "fit-content",
  padding: "14px 18px",
  borderRadius: 18,
  background: "rgba(255,255,255,0.06)",
  border: "1px solid rgba(255,255,255,0.08)",
  color: "#dbe7ef",
  backdropFilter: "blur(14px)",
}

const projectCardStyle = {
  minWidth: 320,
  padding: 20,
  borderRadius: 28,
  background: "linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05))",
  border: "1px solid rgba(255,255,255,0.10)",
  backdropFilter: "blur(14px)",
  boxShadow: "0 24px 52px rgba(0,0,0,0.22)",
}

const projectCardLabelStyle = {
  fontSize: 12,
  color: "#a8bbc7",
  textTransform: "uppercase",
  letterSpacing: "0.12em",
  marginBottom: 10,
}

const projectCardValueStyle = {
  fontSize: 18,
  fontWeight: 700,
  color: "#ffffff",
  marginBottom: 12,
}

const projectCardMetaStyle = {
  fontSize: 14,
  color: "#cad7df",
  lineHeight: 1.8,
}

const projectHealthStyle = {
  marginTop: 12,
  display: "inline-flex",
  padding: "8px 12px",
  borderRadius: 999,
  background: "rgba(87,181,216,0.12)",
  border: "1px solid rgba(87,181,216,0.18)",
  color: "#c5efff",
  fontSize: 12,
  fontWeight: 700,
}

const mainGridStyle = {
  display: "grid",
  gridTemplateColumns: "360px minmax(0, 1fr) 320px",
  gap: 22,
  alignItems: "start",
}

const mainGridMediumStyle = {
  gridTemplateColumns: "320px minmax(0, 1fr)",
}

const mainGridNarrowStyle = {
  gridTemplateColumns: "1fr",
}

const sidebarCardStyle = {
  padding: 20,
  borderRadius: 28,
  background: "linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.035))",
  border: "1px solid rgba(255,255,255,0.09)",
  backdropFilter: "blur(16px)",
  boxShadow: "0 22px 54px rgba(0,0,0,0.18), 0 0 0 1px rgba(106,149,255,0.04)",
}

const sectionTitleStyle = {
  fontWeight: 800,
  fontSize: 18,
  marginBottom: 16,
  color: "#f7fafc",
}

const controlBlockStyle = {
  padding: 16,
  borderRadius: 20,
  background: "rgba(8,12,22,0.34)",
  border: "1px solid rgba(255,255,255,0.06)",
}

const controlSectionStyle = {
  marginBottom: 14,
  padding: 14,
  borderRadius: 22,
  background: "linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02))",
  border: "1px solid rgba(255,255,255,0.06)",
  boxShadow: "inset 0 1px 0 rgba(255,255,255,0.03)",
}

const collapsibleHeaderStyle = {
  width: "100%",
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: 12,
  marginBottom: 14,
  padding: "10px 12px",
  border: "1px solid rgba(255,255,255,0.06)",
  borderRadius: 16,
  background: "rgba(255,255,255,0.04)",
  color: "#f5f8ff",
  cursor: "pointer",
  fontSize: 13,
  fontWeight: 800,
  letterSpacing: "0.04em",
  textTransform: "uppercase",
}

const inputLabelStyle = {
  fontSize: 13,
  fontWeight: 700,
  color: "#d7e4eb",
  marginBottom: 10,
  letterSpacing: "0.03em",
}

const fileInputStyle = {
  width: "100%",
  marginBottom: 12,
  color: "#d9e4eb",
}

const hiddenFileInputStyle = {
  display: "none",
}

const dropzoneStyle = {
  display: "block",
  marginBottom: 14,
  padding: 18,
  borderRadius: 20,
  border: "1px dashed rgba(154,194,215,0.28)",
  background: "linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.025))",
  cursor: "pointer",
}

const dropzoneActiveStyle = {
  border: "1px dashed rgba(93,210,255,0.68)",
  background: "linear-gradient(180deg, rgba(118,97,255,0.16), rgba(34,211,238,0.08))",
  boxShadow: "0 0 0 1px rgba(93,210,255,0.2), 0 20px 42px rgba(20,34,74,0.2)",
  transform: "translateY(-1px)",
}

const dropzoneTitleStyle = {
  color: "#f4f8fb",
  fontWeight: 700,
  marginBottom: 6,
}

const dropzoneTextStyle = {
  color: "#9fb2bf",
  fontSize: 13,
}

const helperTextStyle = {
  marginTop: 10,
  color: "#8ea3b2",
  fontSize: 12,
  lineHeight: 1.55,
}

const authInputStyle = {
  width: "100%",
  padding: "10px 12px",
  borderRadius: 12,
  border: "1px solid rgba(255,255,255,0.14)",
  background: "rgba(255,255,255,0.06)",
  color: "#eef6fb",
}

const validationNoteStyle = {
  marginTop: 12,
  padding: "10px 12px",
  borderRadius: 14,
  fontSize: 12,
  lineHeight: 1.5,
  border: "1px solid rgba(255,255,255,0.08)",
}

const validationErrorStyle = {
  color: "#ffd5d5",
  background: "rgba(127,29,29,0.24)",
  border: "1px solid rgba(248,113,113,0.22)",
}

const validationSuccessStyle = {
  color: "#d7ffe9",
  background: "rgba(6,95,70,0.22)",
  border: "1px solid rgba(52,211,153,0.2)",
}

const validationInfoStyle = {
  color: "#d9f4ff",
  background: "rgba(17,94,89,0.18)",
  border: "1px solid rgba(103,232,249,0.18)",
}

const guidancePanelStyle = {
  marginTop: 14,
  padding: 14,
  borderRadius: 16,
  background: "linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02))",
  border: "1px solid rgba(255,255,255,0.07)",
}

const guidanceTitleStyle = {
  color: "#eef6ff",
  fontSize: 12,
  fontWeight: 800,
  letterSpacing: "0.06em",
  textTransform: "uppercase",
  marginBottom: 10,
}

const guidanceListStyle = {
  display: "grid",
  gap: 8,
}

const guidanceItemStyle = {
  color: "#abc0cf",
  fontSize: 12,
  lineHeight: 1.55,
}

const fileListStyle = {
  marginTop: 12,
  display: "flex",
  flexWrap: "wrap",
  gap: 8,
}

const fileChipStyle = {
  padding: "7px 10px",
  borderRadius: 999,
  background: "rgba(255,255,255,0.07)",
  border: "1px solid rgba(255,255,255,0.08)",
  color: "#d9e8ef",
  fontSize: 12,
}

const actionRowStyle = {
  display: "flex",
  gap: 10,
  flexWrap: "wrap",
}

const controlHeaderRowStyle = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  gap: 12,
}

const metricPillStyle = {
  padding: "6px 10px",
  borderRadius: 999,
  background: "rgba(92,140,176,0.18)",
  border: "1px solid rgba(144,188,219,0.22)",
  color: "#d4ebf6",
  fontSize: 12,
  fontWeight: 700,
}

const rangeStyle = {
  width: "100%",
}

const tabGridStyle = {
  display: "flex",
  gap: 8,
  flexWrap: "wrap",
}

const floorCountGridStyle = {
  display: "grid",
  gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
  gap: 8,
  marginBottom: 14,
}

const floorCountButtonStyle = {
  padding: "10px 12px",
  borderRadius: 14,
  border: "1px solid rgba(255,255,255,0.08)",
  cursor: "pointer",
  fontWeight: 800,
  fontSize: 13,
  transition: "all 0.18s ease",
}

const activeFloorCountButtonStyle = {
  color: "#ffffff",
  background: "linear-gradient(135deg,#7c3aed 0%,#2563eb 58%,#06b6d4 100%)",
  boxShadow: "0 12px 24px rgba(73,92,255,0.28)",
}

const inactiveFloorCountButtonStyle = {
  color: "#d9e7ef",
  background: "rgba(255,255,255,0.06)",
}

const tabButtonStyle = {
  padding: "9px 14px",
  border: "none",
  borderRadius: 12,
  cursor: "pointer",
  fontWeight: 700,
  transition: "all 0.18s ease",
}

const activeTabButtonStyle = {
  color: "#ffffff",
  background: "linear-gradient(135deg,#2c83bf,#57b5d8)",
  boxShadow: "0 10px 24px rgba(61,146,193,0.24)",
}

const inactiveTabButtonStyle = {
  color: "#d7e3ea",
  background: "rgba(255,255,255,0.08)",
}

const infoPanelStyle = {
  marginTop: 4,
  padding: 16,
  borderRadius: 20,
  background: "rgba(214,185,131,0.08)",
  border: "1px solid rgba(221,195,148,0.14)",
}

const statsGridStyle = {
  marginTop: 14,
  display: "grid",
  gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
  gap: 10,
}

const statCardStyle = {
  padding: 14,
  borderRadius: 18,
  background: "rgba(255,255,255,0.05)",
  border: "1px solid rgba(255,255,255,0.07)",
}

const statLabelStyle = {
  fontSize: 12,
  color: "#99adbb",
  textTransform: "uppercase",
  letterSpacing: "0.1em",
  marginBottom: 8,
}

const statValueStyle = {
  fontSize: 24,
  color: "#f8fafb",
  fontWeight: 800,
}

const progressPanelStyle = {
  padding: 16,
  borderRadius: 20,
  background: "rgba(255,255,255,0.05)",
  border: "1px solid rgba(255,255,255,0.07)",
}

const progressRowStyle = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  gap: 10,
  color: "#d7e1e8",
  padding: "8px 0",
  borderTop: "1px solid rgba(255,255,255,0.05)",
}

const progressBadgeStyle = {
  padding: "4px 10px",
  borderRadius: 999,
  background: "rgba(87,181,216,0.14)",
  border: "1px solid rgba(87,181,216,0.2)",
  color: "#bfeeff",
  fontSize: 12,
  fontWeight: 700,
}

const debugPanelStyle = {
  padding: 16,
  borderRadius: 20,
  background: "rgba(255,255,255,0.04)",
  border: "1px solid rgba(255,255,255,0.07)",
}

const processPanelStyle = {
  padding: 16,
  borderRadius: 20,
  background: "rgba(216,235,247,0.06)",
  border: "1px solid rgba(216,235,247,0.09)",
}

const processStepStyle = {
  color: "#d5e3eb",
  fontSize: 13,
  padding: "7px 0",
  borderTop: "1px solid rgba(255,255,255,0.05)",
}

const infoTitleStyle = {
  fontWeight: 700,
  color: "#f2e6cf",
  marginBottom: 10,
}

const infoTextStyle = {
  fontSize: 13,
  color: "#d8d8d3",
  lineHeight: 1.75,
  marginBottom: 6,
}

const errorCardStyle = {
  marginTop: 16,
  padding: 14,
  borderRadius: 16,
  background: "rgba(239,68,68,0.16)",
  border: "1px solid rgba(239,68,68,0.30)",
  color: "#fecaca",
  fontSize: 14,
}

const loadingInlineStyle = {
  marginTop: 16,
  display: "flex",
  alignItems: "center",
  gap: 12,
  color: "#a9d8ef",
  fontSize: 14,
  fontWeight: 600,
}

const viewerColumnStyle = {
  display: "grid",
  gridTemplateColumns: "minmax(320px, 0.82fr) minmax(560px, 1.4fr)",
  gap: 22,
}

const workspaceColumnStyle = {
  display: "grid",
  gap: 22,
  alignContent: "start",
}

const rightRailStyle = {
  display: "grid",
  gap: 14,
  alignContent: "start",
}

const actionRailStyle = {
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: 10,
}

const commonPanelStyle = {
  borderRadius: 28,
  overflow: "hidden",
  background: "linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.035))",
  border: "1px solid rgba(255,255,255,0.09)",
  backdropFilter: "blur(16px)",
  boxShadow: "0 22px 54px rgba(0,0,0,0.18), 0 0 0 1px rgba(124,58,237,0.05)",
}

const planCardStyle = {
  ...commonPanelStyle,
}

const viewerCardStyle = {
  ...commonPanelStyle,
}

const planModeTabsStyle = {
  display: "flex",
  gap: 8,
  flexWrap: "wrap",
  justifyContent: "flex-end",
}

const miniTabStyle = {
  padding: "8px 10px",
  borderRadius: 12,
  border: "1px solid rgba(255,255,255,0.08)",
  cursor: "pointer",
  fontSize: 12,
  fontWeight: 700,
}

const miniTabActiveStyle = {
  background: "linear-gradient(135deg,#2f7db3,#63bddf)",
  color: "#ffffff",
}

const miniTabInactiveStyle = {
  background: "rgba(255,255,255,0.06)",
  color: "#d7e6ee",
}

const panelHeaderStyle = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  padding: "18px 20px",
  borderBottom: "1px solid rgba(255,255,255,0.08)",
}

const panelEyebrowStyle = {
  color: "#9cb6c6",
  fontSize: 11,
  fontWeight: 700,
  letterSpacing: "0.12em",
  textTransform: "uppercase",
  marginBottom: 6,
}

const panelTitleStyle = {
  fontWeight: 800,
  fontSize: 18,
  color: "#f7fafc",
}

const panelBodyStyle = {
  padding: 16,
}

const planImageStyle = {
  width: "100%",
  height: 500,
  objectFit: "contain",
  background: "#ffffff",
  borderRadius: 22,
  boxShadow: "inset 0 0 0 1px rgba(15,23,32,0.08)",
}

const planStageStyle = {
  position: "relative",
  overflow: "hidden",
  borderRadius: 22,
}

const planOverlayStyle = {
  position: "absolute",
  inset: 0,
  width: "100%",
  height: "100%",
  pointerEvents: "none",
}

const legendStyle = {
  position: "absolute",
  left: 16,
  bottom: 16,
  display: "flex",
  flexWrap: "wrap",
  gap: 8,
  padding: "10px 12px",
  borderRadius: 16,
  background: "rgba(10,16,24,0.62)",
  border: "1px solid rgba(255,255,255,0.08)",
  backdropFilter: "blur(12px)",
}

const legendItemStyle = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  color: "#eef5f9",
  fontSize: 12,
}

const legendSwatchStyle = {
  width: 12,
  height: 12,
  borderRadius: 999,
}

const emptyPanelStyle = {
  height: 500,
  display: "grid",
  placeItems: "center",
  color: "#cbd5de",
}

const canvasWrapStyle = {
  padding: 14,
  position: "relative",
}

const canvasStyle = {
  width: "100%",
  height: 860,
  display: "block",
  borderRadius: 22,
  background: "linear-gradient(180deg,#ecf2fb 0%,#d9e4f1 100%)",
  boxShadow:
    "0 0 0 1px rgba(255,255,255,0.08), 0 30px 60px rgba(33,56,120,0.16), 0 0 42px rgba(76,131,255,0.12)",
}

const canvasHintStyle = {
  position: "absolute",
  left: 28,
  bottom: 28,
  padding: "10px 14px",
  borderRadius: 14,
  background: "rgba(7,12,22,0.62)",
  border: "1px solid rgba(255,255,255,0.08)",
  color: "#d9e9ff",
  fontSize: 12,
  fontWeight: 700,
  backdropFilter: "blur(12px)",
}

const buttonStylePrimary = {
  padding: "12px 16px",
  background: "linear-gradient(135deg,#7c3aed 0%,#2563eb 52%,#06b6d4 100%)",
  color: "#ffffff",
  border: "none",
  borderRadius: 14,
  cursor: "pointer",
  fontWeight: 700,
  boxShadow: "0 14px 30px rgba(59,130,246,0.28), 0 0 18px rgba(124,58,237,0.18)",
  transition: "transform 0.18s ease, box-shadow 0.18s ease",
}

const buttonStyleSecondary = {
  padding: "12px 16px",
  background: "linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04))",
  color: "#f2f7fb",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 14,
  cursor: "pointer",
  fontWeight: 700,
}

const buttonStyleGhost = {
  padding: "12px 16px",
  background: "linear-gradient(135deg, rgba(124,58,237,0.16), rgba(34,211,238,0.14))",
  color: "#e6f6ff",
  border: "1px solid rgba(120,162,255,0.24)",
  borderRadius: 14,
  cursor: "pointer",
  fontWeight: 700,
}

const viewerActionRowStyle = {
  display: "flex",
  gap: 8,
}

const viewerActionButtonStyle = {
  padding: "9px 12px",
  borderRadius: 12,
  border: "1px solid rgba(255,255,255,0.08)",
  background: "rgba(255,255,255,0.05)",
  color: "#eef5f9",
  cursor: "pointer",
  fontWeight: 700,
}

const loadingDotStyle = {
  width: 10,
  height: 10,
  borderRadius: "50%",
  background: "#7ed0ef",
  boxShadow: "0 0 0 8px rgba(126,208,239,0.12)",
}

const toastStackStyle = {
  position: "fixed",
  top: 24,
  right: 24,
  zIndex: 10,
  display: "flex",
  flexDirection: "column",
  gap: 10,
}

const toastStyle = {
  padding: "12px 14px",
  borderRadius: 16,
  background: "rgba(14,20,27,0.82)",
  border: "1px solid rgba(255,255,255,0.08)",
  color: "#eef5f9",
  boxShadow: "0 18px 36px rgba(0,0,0,0.18)",
  backdropFilter: "blur(14px)",
}

const toastSuccessStyle = {
  border: "1px solid rgba(109,203,170,0.28)",
}

const toastErrorStyle = {
  border: "1px solid rgba(239,68,68,0.28)",
}

const devDebugPanelStyle = {
  position: "fixed",
  right: 20,
  bottom: 20,
  zIndex: 9,
  width: 260,
  padding: "12px 14px",
  borderRadius: 16,
  background: "rgba(10,16,24,0.86)",
  border: "1px solid rgba(255,255,255,0.08)",
  boxShadow: "0 18px 36px rgba(0,0,0,0.18)",
  backdropFilter: "blur(12px)",
}

const devDebugPanelTitleStyle = {
  color: "#f4f8fb",
  fontSize: 12,
  fontWeight: 800,
  letterSpacing: "0.08em",
  textTransform: "uppercase",
  marginBottom: 8,
}

const devDebugPanelTextStyle = {
  color: "#b8c8d2",
  fontSize: 11,
  lineHeight: 1.55,
  wordBreak: "break-word",
}

export default App


import {
  ArcRotateCamera,
  Color3,
  Color4,
  DirectionalLight,
  DynamicTexture,
  Engine,
  HemisphericLight,
  Mesh,
  MeshBuilder,
  NullEngine,
  PBRMaterial,
  Scene,
  ShadowGenerator,
  StandardMaterial,
  Texture,
  TransformNode,
  Vector3,
  ImageProcessingConfiguration,
} from "@babylonjs/core"
import { CubeTexture } from "@babylonjs/core/Materials/Textures/cubeTexture.js"
import { DefaultRenderingPipeline } from "@babylonjs/core/PostProcesses/RenderPipeline/Pipelines/defaultRenderingPipeline.js"
import { SSAO2RenderingPipeline } from "@babylonjs/core/PostProcesses/RenderPipeline/Pipelines/ssao2RenderingPipeline.js"
import { GLTF2Export } from "@babylonjs/serializers/glTF/index.js"
import earcut from "earcut"
import {
  buildWallGraph,
  classifyOpenings,
  matchOpeningsToWalls,
  splitWallsByOpenings,
} from "./wallGraph.js"

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

function loadTexture(scene, url, options = {}) {
  const texture = new Texture(url, scene, false, true, Texture.TRILINEAR_SAMPLINGMODE)
  texture.uScale = options.uScale ?? 1
  texture.vScale = options.vScale ?? 1
  texture.level = options.level ?? 1
  texture.hasAlpha = options.hasAlpha ?? false
  return texture
}

function createPbrMaterial(scene, name, config) {
  const material = new PBRMaterial(name, scene)
  material.albedoColor = Color3.FromHexString(config.albedoColor ?? "#ffffff")
  material.roughness = config.roughness ?? 0.7
  material.metallic = config.metallic ?? 0
  material.microSurface = config.microSurface ?? 0.82
  material.environmentIntensity = config.environmentIntensity ?? 1
  material.useRoughnessFromMetallicTextureAlpha = false
  material.useRoughnessFromMetallicTextureGreen = true
  material.useMetallnessFromMetallicTextureBlue = false

  if (config.albedoTexture) material.albedoTexture = loadTexture(scene, config.albedoTexture, config.albedoTextureOptions)
  if (config.bumpTexture) material.bumpTexture = loadTexture(scene, config.bumpTexture, config.bumpTextureOptions)
  if (config.ambientTexture) material.ambientTexture = loadTexture(scene, config.ambientTexture, config.ambientTextureOptions)
  if (config.metallicTexture) material.metallicTexture = loadTexture(scene, config.metallicTexture, config.metallicTextureOptions)

  if (config.useAmbientInGrayScale !== undefined) material.useAmbientInGrayScale = config.useAmbientInGrayScale
  if (config.bumpTexture) material.invertNormalMapX = false
  if (config.bumpTexture) material.invertNormalMapY = true

  return material
}

function clonePbrMaterial(baseMaterial, name, overrides = {}) {
  const material = baseMaterial.clone(name)
  if (!material) return baseMaterial

  Object.entries(overrides).forEach(([key, value]) => {
    material[key] = value
  })

  return material
}

function createGlassMaterial(scene, environmentTexture) {
  const material = new PBRMaterial("window-glass", scene)
  material.albedoColor = Color3.FromHexString("#dbe8f2")
  material.metallic = 0
  material.roughness = 0.015
  material.microSurface = 1
  material.alpha = 0.2
  material.indexOfRefraction = 1.5
  material.subSurface.isRefractionEnabled = true
  material.subSurface.indexOfRefraction = 1.52
  material.subSurface.tintColor = Color3.FromHexString("#f7fbff")
  material.subSurface.tintColorAtDistance = 4
  material.environmentIntensity = 2.45
  material.clearCoat.isEnabled = true
  material.clearCoat.intensity = 0.92
  material.clearCoat.roughness = 0.03
  material.emissiveColor = Color3.FromHexString("#0f1318")
  if (environmentTexture) {
    material.reflectionTexture = environmentTexture
  }
  return material
}

function createLandscapeMaterial(scene) {
  return createPbrMaterial(scene, "landscape-ground", {
    albedoColor: "#d4ccb9",
    roughness: 0.97,
    metallic: 0.01,
    environmentIntensity: 0.42,
  })
}

function createParquetAlbedoTexture(scene) {
  const texture = new DynamicTexture("interior-parquet-albedo", { width: 1024, height: 1024 }, scene, false)
  const ctx = texture.getContext()
  const width = 1024
  const height = 1024

  ctx.fillStyle = "#e3d5bf"
  ctx.fillRect(0, 0, width, height)

  const plankCount = 16
  const plankWidth = width / plankCount

  for (let i = 0; i < plankCount; i += 1) {
    const x = i * plankWidth
    const tone = 206 + ((i % 4) * 7 - (i % 2) * 4)
    ctx.fillStyle = `rgb(${tone},${tone - 14},${tone - 28})`
    ctx.fillRect(x, 0, plankWidth, height)

    ctx.fillStyle = "rgba(107,82,56,0.13)"
    ctx.fillRect(x, 0, 2, height)

    for (let y = 0; y < height; y += 92) {
      ctx.fillStyle = `rgba(116,86,58,${0.08 + ((y / 92) % 3) * 0.02})`
      ctx.fillRect(x + 6, y, Math.max(6, plankWidth - 12), 2)
    }

    for (let k = 0; k < 14; k += 1) {
      const grainY = Math.random() * height
      const grainW = 20 + Math.random() * 38
      ctx.fillStyle = `rgba(138,104,72,${0.05 + Math.random() * 0.04})`
      ctx.fillRect(x + 4 + Math.random() * Math.max(2, plankWidth - 16), grainY, grainW, 1.4)
    }
  }

  texture.uScale = 3.8
  texture.vScale = 3.8
  texture.level = 1
  texture.update()
  return texture
}

function createProceduralTexture(scene, name, draw, { uScale = 1, vScale = 1, level = 1 } = {}) {
  const texture = new DynamicTexture(name, { width: 1024, height: 1024 }, scene, false)
  const ctx = texture.getContext()
  draw(ctx, 1024, 1024)
  texture.uScale = uScale
  texture.vScale = vScale
  texture.level = level
  texture.update()
  return texture
}

function createPlasterAlbedoTexture(scene) {
  return createProceduralTexture(scene, "plaster-albedo", (ctx, width, height) => {
    const gradient = ctx.createLinearGradient(0, 0, width, height)
    gradient.addColorStop(0, "#e8e1d6")
    gradient.addColorStop(0.5, "#ddd3c7")
    gradient.addColorStop(1, "#f1ebe3")
    ctx.fillStyle = gradient
    ctx.fillRect(0, 0, width, height)

    for (let i = 0; i < 1800; i += 1) {
      const alpha = 0.012 + Math.random() * 0.02
      const gray = 202 + Math.floor(Math.random() * 18)
      ctx.fillStyle = `rgba(${gray},${gray - 5},${gray - 10},${alpha})`
      const size = 1 + Math.random() * 3
      ctx.fillRect(Math.random() * width, Math.random() * height, size, size)
    }
  }, { uScale: 4.2, vScale: 4.2, level: 0.95 })
}

function createConcreteAlbedoTexture(scene, tone = "#cac2b6") {
  return createProceduralTexture(scene, "concrete-albedo", (ctx, width, height) => {
    ctx.fillStyle = tone
    ctx.fillRect(0, 0, width, height)

    for (let i = 0; i < 2500; i += 1) {
      const alpha = 0.018 + Math.random() * 0.045
      const shade = 156 + Math.floor(Math.random() * 52)
      ctx.fillStyle = `rgba(${shade},${shade - 4},${shade - 8},${alpha})`
      const w = 1 + Math.random() * 6
      const h = 1 + Math.random() * 6
      ctx.fillRect(Math.random() * width, Math.random() * height, w, h)
    }
  }, { uScale: 5.2, vScale: 5.2, level: 1 })
}

function createWoodAlbedoTexture(scene, { base = "#9b6a44", seam = "#6a442b", name = "wood-albedo", scale = 1.8 } = {}) {
  return createProceduralTexture(scene, name, (ctx, width, height) => {
    ctx.fillStyle = base
    ctx.fillRect(0, 0, width, height)

    const plankCount = 12
    const plankWidth = width / plankCount
    for (let i = 0; i < plankCount; i += 1) {
      const x = i * plankWidth
      const tone = 122 + (i % 4) * 9 - (i % 2) * 6
      ctx.fillStyle = `rgb(${tone},${tone - 26},${tone - 44})`
      ctx.fillRect(x, 0, plankWidth, height)

      ctx.fillStyle = "rgba(72,47,30,0.2)"
      ctx.fillRect(x, 0, 2, height)

      for (let k = 0; k < 22; k += 1) {
        const y = Math.random() * height
        const len = 22 + Math.random() * 48
        ctx.fillStyle = `rgba(82,55,35,${0.06 + Math.random() * 0.06})`
        ctx.fillRect(x + 3 + Math.random() * Math.max(2, plankWidth - 12), y, len, 1.2)
      }

      for (let y = 0; y < height; y += 96) {
        ctx.fillStyle = "rgba(95,63,40,0.1)"
        ctx.fillRect(x + 6, y, Math.max(8, plankWidth - 12), 2)
      }
    }

    ctx.fillStyle = seam
    ctx.fillRect(0, 0, width, 2)
  }, { uScale: scale, vScale: scale, level: 1 })
}

function createBrushedMetalTexture(scene) {
  return createProceduralTexture(scene, "metal-brushed", (ctx, width, height) => {
    ctx.fillStyle = "#3b424b"
    ctx.fillRect(0, 0, width, height)

    for (let i = 0; i < 4200; i += 1) {
      const alpha = 0.04 + Math.random() * 0.08
      const val = 122 + Math.floor(Math.random() * 66)
      ctx.fillStyle = `rgba(${val},${val},${val + 2},${alpha})`
      ctx.fillRect(Math.random() * width, Math.random() * height, 10 + Math.random() * 26, 1)
    }
  }, { uScale: 1.6, vScale: 1.6, level: 0.9 })
}

function createMaterialLibrary(scene, environmentTexture) {
  const parquetTexture = createParquetAlbedoTexture(scene)
  const plasterTexture = createPlasterAlbedoTexture(scene)
  const concreteTexture = createConcreteAlbedoTexture(scene, "#c8bfb2")
  const pavingTexture = createConcreteAlbedoTexture(scene, "#bfb6a9")
  const roofTexture = createConcreteAlbedoTexture(scene, "#b8bcc1")
  const woodTexture = createWoodAlbedoTexture(scene, { base: "#916241", seam: "#60412a", name: "door-wood", scale: 1.5 })
  const metalTexture = createBrushedMetalTexture(scene)
  const slabMaterial = createPbrMaterial(scene, "slab", {
    albedoColor: "#eadbc4",
    roughness: 0.62,
    metallic: 0.02,
    microSurface: 0.84,
    environmentIntensity: 0.82,
  })
  slabMaterial.albedoTexture = parquetTexture
  const debugSolid = new StandardMaterial("debug-solid", scene)
  debugSolid.diffuseColor = Color3.FromHexString("#ff6f3c")
  debugSolid.emissiveColor = Color3.FromHexString("#2a1206")
  const fallback = new StandardMaterial("fallback-gray", scene)
  fallback.diffuseColor = Color3.FromHexString("#b7b7b7")

  const materials = {
    exteriorWall: createPbrMaterial(scene, "exterior-wall", {
      albedoColor: "#e2ddd5",
      roughness: 0.82,
      metallic: 0.01,
      environmentIntensity: 0.88,
    }),
    innerWall: createPbrMaterial(scene, "inner-wall", {
      albedoColor: "#fbfaf7",
      roughness: 0.93,
      metallic: 0.0,
      environmentIntensity: 0.72,
    }),
    innerTrim: createPbrMaterial(scene, "inner-trim", {
      albedoColor: "#f2eee7",
      roughness: 0.92,
      metallic: 0.0,
      environmentIntensity: 0.62,
    }),
    floorBand: createPbrMaterial(scene, "floor-band", {
      albedoColor: "#b3aaa0",
      roughness: 0.84,
      metallic: 0.04,
      environmentIntensity: 0.74,
    }),
    plinth: createPbrMaterial(scene, "plinth", {
      albedoColor: "#c7beb2",
      roughness: 0.94,
      metallic: 0.02,
      environmentIntensity: 0.58,
    }),
    slab: slabMaterial,
    roof: createPbrMaterial(scene, "roof", {
      albedoColor: "#b9bcc1",
      roughness: 0.88,
      metallic: 0.02,
      environmentIntensity: 0.74,
    }),
    metal: createPbrMaterial(scene, "metal", {
      albedoColor: "#353c44",
      roughness: 0.42,
      metallic: 0.78,
      environmentIntensity: 1.2,
    }),
    facadeAccent: createPbrMaterial(scene, "facade-accent", {
      albedoColor: "#c4bbb0",
      roughness: 0.81,
      metallic: 0.02,
      environmentIntensity: 0.92,
    }),
    interiorGlow: (() => {
      const m = new StandardMaterial("interior-glow", scene)
      m.diffuseColor = Color3.FromHexString("#d9bb95")
      m.emissiveColor = Color3.FromHexString("#8c6240")
      m.specularColor = Color3.Black()
      return m
    })(),
    wood: createPbrMaterial(scene, "wood", {
      albedoColor: "#9b6a44",
      roughness: 0.52,
      metallic: 0.06,
      environmentIntensity: 0.7,
    }),
    glass: createGlassMaterial(scene, environmentTexture),
    paving: createPbrMaterial(scene, "paving", {
      albedoColor: "#c2bbb0",
      roughness: 0.88,
      metallic: 0.01,
      environmentIntensity: 0.65,
    }),
    landscape: createLandscapeMaterial(scene),
    siteLawn: createPbrMaterial(scene, "site-lawn", {
      albedoColor: "#8ea678",
      roughness: 0.95,
      metallic: 0,
      environmentIntensity: 0.42,
    }),
    sitePlant: createPbrMaterial(scene, "site-plant", {
      albedoColor: "#6f8660",
      roughness: 0.9,
      metallic: 0,
      environmentIntensity: 0.34,
    }),
    siteRoad: createPbrMaterial(scene, "site-road", {
      albedoColor: "#7f8082",
      roughness: 0.92,
      metallic: 0.01,
      environmentIntensity: 0.36,
    }),
    fallback,
    debugSolid,
  }

  materials.exteriorWall.albedoTexture = plasterTexture
  materials.innerWall.albedoTexture = plasterTexture
  materials.innerTrim.albedoTexture = plasterTexture
  materials.plinth.albedoTexture = concreteTexture
  materials.roof.albedoTexture = roofTexture
  materials.wood.albedoTexture = woodTexture
  materials.metal.albedoTexture = metalTexture
  materials.paving.albedoTexture = pavingTexture
  materials.siteRoad.albedoTexture = pavingTexture

  return materials
}

function createExportMaterialLibrary(scene) {
  const fallback = new StandardMaterial("export-fallback", scene)
  fallback.diffuseColor = Color3.FromHexString("#c8c3bb")

  const wall = new StandardMaterial("export-wall", scene)
  wall.diffuseColor = Color3.FromHexString("#d9d3ca")

  const slab = new StandardMaterial("export-slab", scene)
  slab.diffuseColor = Color3.FromHexString("#c6c0b6")

  const roof = new StandardMaterial("export-roof", scene)
  roof.diffuseColor = Color3.FromHexString("#50545a")

  const metal = new StandardMaterial("export-metal", scene)
  metal.diffuseColor = Color3.FromHexString("#7a7d82")

  const wood = new StandardMaterial("export-wood", scene)
  wood.diffuseColor = Color3.FromHexString("#8d5c3f")

  const glass = new StandardMaterial("export-glass", scene)
  glass.diffuseColor = Color3.FromHexString("#c5d7e3")
  glass.alpha = 0.45

  return {
    exteriorWall: wall,
    innerWall: wall,
    plinth: slab,
    slab,
    roof,
    metal,
    wood,
    glass,
    paving: slab,
    fallback,
    debugSolid: fallback,
  }
}

function setupEnvironment(scene, isBuildingView = false) {
  scene.clearColor = new Color4(0.86, 0.91, 0.97, 1)

  let environmentTexture = null

  try {
    environmentTexture = CubeTexture.CreateFromPrefilteredData("/hdr/studio_small_01_4k.env", scene)
    if (environmentTexture) {
      scene.environmentTexture = environmentTexture
      scene.environmentIntensity = isBuildingView ? 1.35 : 1.12
      scene.createDefaultSkybox(environmentTexture, true, 1000)
    }
  } catch (error) {
    console.warn("HDRI .env could not be loaded, continuing without environment.", error)
  }

  const skyTexture = new DynamicTexture("sky-gradient", { width: 1024, height: 1024 }, scene, false)
  const ctx = skyTexture.getContext()
  const gradient = ctx.createLinearGradient(0, 0, 0, 1024)
  gradient.addColorStop(0, isBuildingView ? "#6fa5d6" : "#8ec0eb")
  gradient.addColorStop(0.44, isBuildingView ? "#9dc5e3" : "#bddbf2")
  gradient.addColorStop(0.78, "#edf5fb")
  gradient.addColorStop(1, "#f8fbff")
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, 1024, 1024)

  const sunGlow = ctx.createRadialGradient(780, 220, 40, 780, 220, 220)
  sunGlow.addColorStop(0, "rgba(255,245,220,0.38)")
  sunGlow.addColorStop(1, "rgba(255,245,220,0)")
  ctx.fillStyle = sunGlow
  ctx.fillRect(0, 0, 1024, 1024)
  skyTexture.update()

  const skybox = MeshBuilder.CreateBox("skybox", { size: 850 }, scene)
  const skyboxMaterial = new StandardMaterial("skybox-material", scene)
  skyboxMaterial.backFaceCulling = false
  skyboxMaterial.disableLighting = true
  skyboxMaterial.emissiveTexture = skyTexture
  skyboxMaterial.diffuseTexture = skyTexture
  skyboxMaterial.specularColor = Color3.Black()
  skybox.material = skyboxMaterial
  skybox.infiniteDistance = true

  return environmentTexture
}

function setupBuildingPostFx(scene, camera) {
  let pipeline = null
  let ssao = null

  try {
    pipeline = new DefaultRenderingPipeline("building-pipeline", true, scene, [camera])
    pipeline.samples = 1
    pipeline.fxaaEnabled = true
    pipeline.imageProcessingEnabled = true
    pipeline.chromaticAberrationEnabled = false
    pipeline.sharpenEnabled = false

    ssao = new SSAO2RenderingPipeline("building-ssao", scene, {
      ssaoRatio: 0.5,
      blurRatio: 1,
    }, [camera])
    ssao.totalStrength = 0.65
    ssao.radius = 1.05
    ssao.maxZ = 120
    ssao.expensiveBlur = false
    scene.postProcessRenderPipelineManager.attachCamerasToRenderPipeline("building-ssao", camera)
  } catch (error) {
    console.warn("Building post FX could not be fully enabled.", error)
  }

  return () => {
    try {
      if (ssao) {
        scene.postProcessRenderPipelineManager.detachCamerasFromRenderPipeline("building-ssao", camera)
      }
      ssao?.dispose?.()
      pipeline?.dispose?.()
    } catch {
      // noop
    }
  }
}

function setupLighting(scene, activeFloor, camera) {
  const isBuildingView = activeFloor === "building"
  const hemi = new HemisphericLight("hemi", new Vector3(0, 1, 0), scene)
  hemi.intensity = isBuildingView ? 0.7 : 0.92
  hemi.groundColor = Color3.FromHexString("#d9cfbe")

  const sun = new DirectionalLight("sun", new Vector3(-0.9, -1.6, -0.45), scene)
  sun.position = new Vector3(30, 34, 18)
  sun.intensity = isBuildingView ? 3.25 : 2.5

  const shadowGenerator = new ShadowGenerator(isBuildingView ? 2048 : 2048, sun)
  shadowGenerator.usePercentageCloserFiltering = true
  shadowGenerator.filteringQuality = ShadowGenerator.QUALITY_HIGH
  shadowGenerator.bias = 0.00008
  shadowGenerator.normalBias = isBuildingView ? 0.018 : 0.015
  shadowGenerator.darkness = isBuildingView ? 0.3 : 0.22

  const cameraFill = new DirectionalLight("camera-fill", new Vector3(0.2, -0.7, 0.2), scene)
  cameraFill.intensity = isBuildingView ? 0.58 : 0.82
  const rim = new DirectionalLight("rim", new Vector3(0.35, -0.5, -0.35), scene)
  rim.intensity = isBuildingView ? 0.42 : 0.22

  const lightObserver = scene.onBeforeRenderObservable.add(() => {
    if (!camera) return
    const target = camera.target ?? Vector3.Zero()
    const dir = target.subtract(camera.position)
    if (dir.lengthSquared() < 0.0001) return
    dir.normalize()
    cameraFill.direction = new Vector3(dir.x, Math.min(-0.45, dir.y - 0.2), dir.z)
    cameraFill.position = camera.position.add(new Vector3(-dir.x * 6.2, 4.2, -dir.z * 6.2))
    rim.direction = new Vector3(-dir.x * 0.8, -0.45, -dir.z * 0.8)
    rim.position = camera.position.add(new Vector3(dir.x * 7.6, 3.8, dir.z * 7.6))
  })

  return {
    shadowGenerator,
    disposeLighting() {
      scene.onBeforeRenderObservable.remove(lightObserver)
      rim.dispose()
      cameraFill.dispose()
      sun.dispose()
      hemi.dispose()
    },
  }
}

function createGround(scene, envelope, materials, shadowGenerator) {
  const width = envelope.maxX - envelope.minX
  const depth = envelope.maxZ - envelope.minZ
  const landscape = MeshBuilder.CreateGround("site-landscape", {
    width: Math.max(30, width * 1.9),
    height: Math.max(30, depth * 1.9),
  }, scene)
  landscape.position.y = -0.03
  landscape.position.x = (envelope.minX + envelope.maxX) / 2
  landscape.position.z = (envelope.minZ + envelope.maxZ) / 2
  landscape.material = materials.landscape || materials.paving
  landscape.receiveShadows = true

  return landscape
}

function createBuildingMaterialVariant(scene, materials, environmentTexture) {
  const enhancedGlass = createGlassMaterial(scene, environmentTexture)
  enhancedGlass.albedoColor = Color3.FromHexString("#a7bac9")
  enhancedGlass.roughness = 0.03
  enhancedGlass.alpha = 0.24
  enhancedGlass.environmentIntensity = 2.9
  enhancedGlass.clearCoat.roughness = 0.02
  enhancedGlass.subSurface.tintColor = Color3.FromHexString("#d6e6f4")

  const buildingFacadeTexture = createPlasterAlbedoTexture(scene)
  buildingFacadeTexture.uScale = 3.4
  buildingFacadeTexture.vScale = 3.4

  const buildingSlabTexture = createParquetAlbedoTexture(scene)
  buildingSlabTexture.uScale = 3.2
  buildingSlabTexture.vScale = 3.2

  const buildingDoorTexture = createWoodAlbedoTexture(scene, {
    base: "#875937",
    seam: "#51331f",
    name: "building-door-wood",
    scale: 1.35,
  })

  const buildingMaterials = {
    ...materials,
    exteriorWall: clonePbrMaterial(materials.exteriorWall, "exterior-wall-building", {
      albedoColor: Color3.FromHexString("#ddd4c7"),
      roughness: 0.74,
      metallic: 0.02,
      microSurface: 0.88,
      environmentIntensity: 1.02,
    }),
    slab: clonePbrMaterial(materials.slab, "slab-building", {
      albedoColor: Color3.FromHexString("#d2c0aa"),
      roughness: 0.66,
      metallic: 0.04,
      microSurface: 0.86,
      environmentIntensity: 0.92,
    }),
    roof: clonePbrMaterial(materials.roof, "roof-building", {
      albedoColor: Color3.FromHexString("#bfc5cb"),
      roughness: 0.78,
      metallic: 0.04,
      microSurface: 0.83,
      environmentIntensity: 0.95,
    }),
    floorBand: clonePbrMaterial(materials.floorBand, "floor-band-building", {
      albedoColor: Color3.FromHexString("#8e857a"),
      roughness: 0.58,
      metallic: 0.12,
      microSurface: 0.76,
      environmentIntensity: 1.04,
    }),
    metal: clonePbrMaterial(materials.metal, "metal-building", {
      albedoColor: Color3.FromHexString("#2f363e"),
      roughness: 0.28,
      metallic: 0.9,
      microSurface: 0.92,
      environmentIntensity: 1.28,
    }),
    wood: clonePbrMaterial(materials.wood, "wood-building", {
      albedoColor: Color3.FromHexString("#845535"),
      roughness: 0.38,
      metallic: 0.03,
      microSurface: 0.84,
      environmentIntensity: 0.92,
    }),
    plinth: clonePbrMaterial(materials.plinth, "plinth-building", {
      albedoColor: Color3.FromHexString("#c5bbb0"),
      roughness: 0.82,
      metallic: 0.03,
      environmentIntensity: 0.82,
    }),
    landscape: clonePbrMaterial(materials.landscape, "landscape-building", {
      albedoColor: Color3.FromHexString("#cec5b5"),
      roughness: 0.95,
      metallic: 0,
      environmentIntensity: 0.55,
    }),
    glass: enhancedGlass,
    windowMetal: materials.metal,
    windowGlass: materials.glass,
    windowSill: materials.slab,
    windowRecess: materials.fallback,
  }

  if (buildingMaterials.exteriorWall) buildingMaterials.exteriorWall.albedoTexture = buildingFacadeTexture
  if (buildingMaterials.slab) buildingMaterials.slab.albedoTexture = buildingSlabTexture
  if (buildingMaterials.wood) buildingMaterials.wood.albedoTexture = buildingDoorTexture

  return buildingMaterials
}

function applyVisualOverrides(materials, overrides = {}) {
  const setAlbedo = (material, hex) => {
    if (!material || !hex || !material.albedoColor) return
    material.albedoColor = Color3.FromHexString(hex)
  }
  const setDiffuse = (material, hex) => {
    if (!material || !hex || !material.diffuseColor) return
    material.diffuseColor = Color3.FromHexString(hex)
  }

  setAlbedo(materials.exteriorWall, overrides.walls)
  setAlbedo(materials.innerWall, overrides.walls)
  setAlbedo(materials.innerTrim, overrides.walls)
  setAlbedo(materials.slab, overrides.slabs)
  setAlbedo(materials.floorBand, overrides.slabs)
  setAlbedo(materials.roof, overrides.roof)
  setAlbedo(materials.metal, overrides.frames)
  setAlbedo(materials.wood, overrides.door)
  setAlbedo(materials.glass, overrides.glass)
  setAlbedo(materials.landscape, overrides.ground)
  setAlbedo(materials.paving, overrides.ground)
  setAlbedo(materials.plinth, overrides.ground)
  setDiffuse(materials.fallback, overrides.ground)
}

function createPodium(scene, envelope, materials, shadowGenerator, parent = null, options = {}) {
  const width = envelope.maxX - envelope.minX
  const depth = envelope.maxZ - envelope.minZ
  const podium = createBox(
    scene,
    "site-podium",
    {
      width: width + 0.18,
      height: 0.14,
      depth: depth + 0.18,
    },
    new Vector3((envelope.minX + envelope.maxX) / 2, 0.06, (envelope.minZ + envelope.maxZ) / 2),
    materials.plinth,
    shadowGenerator,
    parent,
  )

  if (options.includeCap) {
    createBox(
      scene,
      "site-podium-cap",
      {
        width: width + 0.12,
        height: 0.025,
        depth: depth + 0.12,
      },
      new Vector3((envelope.minX + envelope.maxX) / 2, 0.145, (envelope.minZ + envelope.maxZ) / 2),
      materials.slab || materials.plinth,
      shadowGenerator,
      parent,
    )
  }

  return podium
}

function createSimpleTree(scene, parent, shadowGenerator, position, materials) {
  const trunk = MeshBuilder.CreateCylinder("site-tree-trunk", {
    height: 0.82,
    diameterTop: 0.08,
    diameterBottom: 0.1,
  }, scene)
  trunk.position = position.add(new Vector3(0, 0.48, 0))
  trunk.material = materials.wood || materials.metal || materials.fallback
  trunk.receiveShadows = true
  shadowGenerator?.addShadowCaster(trunk)
  trunk.parent = parent

  const crown = MeshBuilder.CreateSphere("site-tree-crown", { diameter: 0.72 }, scene)
  crown.position = position.add(new Vector3(0, 1.02, 0))
  crown.material = materials.sitePlant || materials.siteLawn || materials.fallback
  crown.receiveShadows = true
  shadowGenerator?.addShadowCaster(crown)
  crown.parent = parent
}

function addSiteContext(scene, root, envelope, materials, shadowGenerator) {
  if (!envelope) return

  const width = envelope.maxX - envelope.minX
  const depth = envelope.maxZ - envelope.minZ
  const centerX = (envelope.minX + envelope.maxX) / 2
  const centerZ = (envelope.minZ + envelope.maxZ) / 2
  const frontZ = envelope.maxZ + Math.max(1.9, depth * 0.18)

  createBox(
    scene,
    "site-forecourt",
    {
      width: width + 1.9,
      height: 0.04,
      depth: Math.max(2.7, depth * 0.22),
    },
    new Vector3(centerX, 0.29, frontZ),
    materials.paving || materials.plinth,
    shadowGenerator,
    root,
  )

  const roadZ = frontZ + Math.max(1.25, depth * 0.12)
  createBox(
    scene,
    "site-road",
    {
      width: Math.max(width + 8, 16),
      height: 0.03,
      depth: Math.max(3.6, depth * 0.28),
    },
    new Vector3(centerX, 0.275, roadZ),
    materials.siteRoad || materials.paving || materials.fallback,
    shadowGenerator,
    root,
  )

  createBox(
    scene,
    "site-walkway",
    {
      width: Math.max(1.9, width * 0.18),
      height: 0.042,
      depth: Math.max(3.1, depth * 0.28),
    },
    new Vector3(centerX, 0.292, envelope.maxZ + Math.max(1.3, depth * 0.14)),
    materials.slab || materials.paving,
    shadowGenerator,
    root,
  )

  createBox(
    scene,
    "site-walkway-border-left",
    {
      width: 0.06,
      height: 0.03,
      depth: Math.max(3.2, depth * 0.3),
    },
    new Vector3(centerX - Math.max(0.95, width * 0.09), 0.296, envelope.maxZ + Math.max(1.28, depth * 0.14)),
    materials.facadeAccent || materials.plinth || materials.fallback,
    shadowGenerator,
    root,
  )
  createBox(
    scene,
    "site-walkway-border-right",
    {
      width: 0.06,
      height: 0.03,
      depth: Math.max(3.2, depth * 0.3),
    },
    new Vector3(centerX + Math.max(0.95, width * 0.09), 0.296, envelope.maxZ + Math.max(1.28, depth * 0.14)),
    materials.facadeAccent || materials.plinth || materials.fallback,
    shadowGenerator,
    root,
  )

}

function clusterValues(values, tolerance = 0.55) {
  const sorted = values.slice().sort((a, b) => a - b)
  const groups = []
  sorted.forEach((value) => {
    const last = groups[groups.length - 1]
    if (!last || Math.abs(value - last.center) > tolerance) {
      groups.push({ center: value, count: 1 })
      return
    }
    const nextCount = last.count + 1
    last.center = (last.center * last.count + value) / nextCount
    last.count = nextCount
  })
  return groups
}

function createBalconyAt(scene, floorParent, shadowGenerator, materials, worldX, frontZ, yBase, floorHeight, widthHint) {
  const balconyDepth = 0.74
  const balconyHeight = 0.085
  const balconyWidth = Math.max(1.45, Math.min(widthHint, 3.2))
  const balconyY = yBase + SLAB_THICKNESS + 0.04
  const railingY = yBase + SLAB_THICKNESS + Math.max(0.95, floorHeight * 0.32)
  const slabZ = frontZ + balconyDepth * 0.5 - 0.05
  const frontZEdge = frontZ + balconyDepth - 0.02

  createBox(
    scene,
    `balcony-slab-${worldX}-${yBase}`,
    { width: balconyWidth, height: balconyHeight, depth: balconyDepth },
    new Vector3(worldX, balconyY, slabZ),
    materials.slab || materials.plinth || materials.fallback,
    shadowGenerator,
    floorParent,
  )

  createBox(
    scene,
    `balcony-rail-bottom-${worldX}-${yBase}`,
    { width: balconyWidth - 0.08, height: 0.05, depth: 0.045 },
    new Vector3(worldX, railingY - 0.44, frontZEdge),
    materials.metal || materials.fallback,
    shadowGenerator,
    floorParent,
  )

  const glass = MeshBuilder.CreateBox(`balcony-glass-${worldX}-${yBase}`, {
    width: balconyWidth - 0.12,
    height: 0.9,
    depth: 0.025,
  }, scene)
  glass.position = new Vector3(worldX, railingY, frontZEdge)
  glass.material = materials.glass || materials.fallback
  glass.parent = floorParent
  glass.receiveShadows = true

  createBox(
    scene,
    `balcony-rail-top-${worldX}-${yBase}`,
    { width: balconyWidth - 0.06, height: 0.045, depth: 0.045 },
    new Vector3(worldX, railingY + 0.44, frontZEdge),
    materials.metal || materials.fallback,
    shadowGenerator,
    floorParent,
  )
}

function addBuildingFacadeComposition(
  scene,
  floorParent,
  shadowGenerator,
  materials,
  floorEnvelope,
  matchedOpenings,
  geometryMeta,
  yBase,
  floorHeight,
  floorIndex,
) {
  const frontZ = floorEnvelope.maxZ
  const width = Math.max(1.2, floorEnvelope.maxX - floorEnvelope.minX)
  const centerX = (floorEnvelope.minX + floorEnvelope.maxX) / 2
  const facadeMat = materials.facadeAccent || materials.wood || materials.plinth || materials.fallback
  const beltMat = materials.metal || materials.floorBand || materials.fallback

  const outerWindows = (matchedOpenings || [])
    .filter((opening) => opening.type === "window" && opening.wallKind === "outer")
    .map((opening) => {
      const wp = geometryMeta.transformPoint(opening.point.x, opening.point.y)
      return { ...opening, wp }
    })

  const frontWindows = outerWindows.filter((opening) => Math.abs(opening.wp.z - frontZ) <= 0.55)

  const xColumns = clusterValues(frontWindows.map((opening) => opening.wp.x), 0.72)
    .sort((a, b) => a.center - b.center)

  xColumns.forEach((column, columnIndex) => {
    if (column.count < 1) return
    const spanHeight = Math.max(2.2, floorHeight - 0.1)
    createBox(
      scene,
      `facade-column-${floorIndex}-${columnIndex}`,
      { width: 0.12, height: spanHeight, depth: 0.06 },
      new Vector3(column.center, yBase + SLAB_THICKNESS + spanHeight / 2, frontZ + 0.055),
      facadeMat,
      shadowGenerator,
      floorParent,
    )
  })

  createBox(
    scene,
    `facade-belt-${floorIndex}`,
    { width: Math.max(1.2, width - 0.22), height: 0.06, depth: 0.05 },
    new Vector3(centerX, yBase + floorHeight + SLAB_THICKNESS - 0.08, frontZ + 0.055),
    beltMat,
    shadowGenerator,
    floorParent,
  )

  if (floorIndex === 0) return

  const closestColumns = xColumns
    .slice()
    .sort((a, b) => Math.abs(a.center - centerX) - Math.abs(b.center - centerX))
    .slice(0, 2)

  const derivedCenter = closestColumns.length
    ? closestColumns.reduce((sum, col) => sum + col.center, 0) / closestColumns.length
    : centerX

  const derivedWidth =
    closestColumns.length === 2
      ? Math.abs(closestColumns[0].center - closestColumns[1].center) + 1.15
      : width * 0.34

  createBalconyAt(
    scene,
    floorParent,
    shadowGenerator,
    materials,
    derivedCenter,
    frontZ,
    yBase,
    floorHeight,
    Math.max(1.8, Math.min(3.6, derivedWidth)),
  )
}

function createBox(scene, name, size, position, material, shadowGenerator, parent = null) {
  const mesh = MeshBuilder.CreateBox(name, size, scene)
  mesh.position.copyFrom(position)
  mesh.material = material
  mesh.receiveShadows = true
  shadowGenerator?.addShadowCaster(mesh)
  if (parent) mesh.parent = parent
  return mesh
}

function createWallSegment(scene, parent, shadowGenerator, material, geometryMeta, line, thickness, height, yBase, options = {}) {
  const [x1, y1, x2, y2] = line
  const p1 = geometryMeta.transformPoint(x1, y1)
  const p2 = geometryMeta.transformPoint(x2, y2)
  const dx = p2.x - p1.x
  const dz = p2.z - p1.z
  const length = Math.hypot(dx, dz)
  if (length < 0.08) return null

  const mesh = MeshBuilder.CreateBox("wall-segment", {
    width: length,
    height,
    depth: thickness,
  }, scene)
  mesh.position = new Vector3((p1.x + p2.x) / 2, yBase + height / 2, (p1.z + p2.z) / 2)
  mesh.rotation.y = Math.atan2(dz, dx)
  mesh.material = material
  mesh.receiveShadows = true
  shadowGenerator?.addShadowCaster(mesh)
  mesh.parent = parent

  if (options.isInner && options.trimMaterial) {
    const baseTrim = MeshBuilder.CreateBox("inner-wall-base-trim", {
      width: length,
      height: 0.05,
      depth: thickness + 0.02,
    }, scene)
    baseTrim.position = new Vector3((p1.x + p2.x) / 2, yBase + 0.025, (p1.z + p2.z) / 2)
    baseTrim.rotation = new Vector3(0, Math.atan2(dz, dx), 0)
    baseTrim.material = options.trimMaterial
    baseTrim.receiveShadows = true
    shadowGenerator?.addShadowCaster(baseTrim)
    baseTrim.parent = parent

    const topTrim = MeshBuilder.CreateBox("inner-wall-top-trim", {
      width: length,
      height: 0.04,
      depth: thickness + 0.015,
    }, scene)
    topTrim.position = new Vector3((p1.x + p2.x) / 2, yBase + height - 0.02, (p1.z + p2.z) / 2)
    topTrim.rotation = new Vector3(0, Math.atan2(dz, dx), 0)
    topTrim.material = options.trimMaterial
    topTrim.receiveShadows = true
    shadowGenerator?.addShadowCaster(topTrim)
    topTrim.parent = parent
  }

  return mesh
}

function createBuildingWallSegmentRealistic(
  scene,
  parent,
  shadowGenerator,
  material,
  trimMaterial,
  geometryMeta,
  line,
  thickness,
  height,
  yBase,
) {
  const wall = createWallSegment(
    scene,
    parent,
    shadowGenerator,
    material,
    geometryMeta,
    line,
    thickness,
    height,
    yBase,
    {}
  )
  if (!wall) return null

  const [x1, y1, x2, y2] = line
  const p1 = geometryMeta.transformPoint(x1, y1)
  const p2 = geometryMeta.transformPoint(x2, y2)
  const dx = p2.x - p1.x
  const dz = p2.z - p1.z
  const angle = Math.atan2(dz, dx)
  const length = Math.hypot(dx, dz)
  if (length < 0.1) return wall

  const edgeMat = trimMaterial || material
  const capHeight = 0.04
  const capDepth = Math.max(0.016, thickness * 0.22)

  const baseCap = MeshBuilder.CreateBox("building-wall-base-cap", {
    width: length,
    height: capHeight,
    depth: capDepth,
  }, scene)
  baseCap.position = new Vector3((p1.x + p2.x) / 2, yBase + capHeight * 0.5, (p1.z + p2.z) / 2)
  baseCap.rotation = new Vector3(0, angle, 0)
  baseCap.material = edgeMat
  baseCap.parent = parent
  baseCap.receiveShadows = true
  shadowGenerator?.addShadowCaster(baseCap)

  const topCap = MeshBuilder.CreateBox("building-wall-top-cap", {
    width: length,
    height: capHeight,
    depth: capDepth,
  }, scene)
  topCap.position = new Vector3((p1.x + p2.x) / 2, yBase + height - capHeight * 0.5, (p1.z + p2.z) / 2)
  topCap.rotation = new Vector3(0, angle, 0)
  topCap.material = edgeMat
  topCap.parent = parent
  topCap.receiveShadows = true
  shadowGenerator?.addShadowCaster(topCap)

  return wall
}

function createAlignedWallFill(scene, parent, shadowGenerator, material, geometryMeta, line, centerPoint, widthWorld, yCenter, height, thickness) {
  const [x1, y1, x2, y2] = line
  const p1 = geometryMeta.transformPoint(x1, y1)
  const p2 = geometryMeta.transformPoint(x2, y2)
  const center = geometryMeta.transformPoint(centerPoint.x, centerPoint.y)
  const angle = Math.atan2(p2.z - p1.z, p2.x - p1.x)

  if (widthWorld < 0.06 || height < 0.06) return null

  const fill = MeshBuilder.CreateBox("opening-wall-fill", {
    width: widthWorld,
    height,
    depth: thickness,
  }, scene)
  fill.position = new Vector3(center.x, yCenter, center.z)
  fill.rotation = new Vector3(0, angle, 0)
  fill.material = material
  fill.receiveShadows = true
  shadowGenerator?.addShadowCaster(fill)
  fill.parent = parent
  return fill
}

function createOpeningLookup(matched) {
  const byId = new Map()
  ;(matched || []).forEach((opening) => byId.set(opening.id, opening))
  return byId
}

function resolveSourceOpening(item, openingById) {
  if (!item?.sourceIds?.length) return null
  for (const sourceId of item.sourceIds) {
    const hit = openingById.get(sourceId)
    if (hit) return hit
  }
  return null
}

async function exportProjectGlb(project, geometryMeta, floorHeight, filenameBase) {
  const engine = new NullEngine()
  const scene = new Scene(engine)
  const materials = createExportMaterialLibrary(scene)
  const root = createMinimalBuildingScene(
    scene,
    project,
    geometryMeta,
    "building",
    floorHeight,
    materials,
    null,
  )

  const exportNodeIds = new Set(
    [root, ...(root?.getDescendants?.(false) || [])]
      .filter(Boolean)
      .map((node) => node.uniqueId)
  )

  try {
    const glb = await GLTF2Export.GLBAsync(scene, filenameBase, {
      shouldExportNode: (node) => exportNodeIds.has(node.uniqueId),
    })
    const fileEntries = Object.entries(glb.glTFFiles || {})
    const glbEntry = fileEntries.find(([name]) => name.toLowerCase().endsWith(".glb"))
    if (!glbEntry) {
      throw new Error("GLB blob could not be created")
    }

    const [filename, payload] = glbEntry
    const blob = payload instanceof Blob
      ? payload
      : new Blob([payload], { type: "model/gltf-binary" })

    return { blob, filename }
  } finally {
    scene.dispose()
    engine.dispose()
  }
}

function createWindow(scene, parent, shadowGenerator, materials, geometryMeta, opening, yBase, thickness, options = {}) {
  const [x1, y1, x2, y2] = opening.hostLine
  const p1 = geometryMeta.transformPoint(x1, y1)
  const p2 = geometryMeta.transformPoint(x2, y2)
  const center = geometryMeta.transformPoint(opening.point.x, opening.point.y)
  const angle = Math.atan2(p2.z - p1.z, p2.x - p1.x)
  const width = Math.max(0.92, opening.widthPx * geometryMeta.scale * 0.52)
  const group = new TransformNode("window-group", scene)
  group.parent = parent
  group.position = new Vector3(center.x, yBase, center.z)
  group.rotation = new Vector3(0, angle, 0)

  const frameDepth = Math.max(0.06, thickness * 0.22)
  const frameThickness = 0.05
  const paneDepth = Math.max(0.03, thickness * 0.06)
  const facadeOffset = thickness * 0.62
  const windowMetal = materials.windowMetal || materials.metal
  const windowGlass = materials.windowGlass || materials.glass
  const windowSill = materials.windowSill || materials.slab
  const windowRecess = materials.windowRecess || materials.fallback

  ;[-1, 1].forEach((side) => {
    const z = facadeOffset * side

    const recess = MeshBuilder.CreateBox(`window-recess-${side}`, {
      width: width + 0.1,
      height: WINDOW_HEIGHT + 0.18,
      depth: Math.max(0.06, thickness * 0.16),
    }, scene)
    recess.position = new Vector3(0, WINDOW_SILL_HEIGHT + WINDOW_HEIGHT / 2, z - side * 0.02)
    recess.material = windowRecess
    recess.visibility = 0.22
    recess.parent = group

    createBox(scene, `window-frame-top-${side}`, {
      width,
      height: frameThickness,
      depth: frameDepth,
    }, new Vector3(0, WINDOW_SILL_HEIGHT + WINDOW_HEIGHT - frameThickness / 2, z), windowMetal, shadowGenerator, group)
    createBox(scene, `window-frame-bottom-${side}`, {
      width,
      height: frameThickness,
      depth: frameDepth,
    }, new Vector3(0, WINDOW_SILL_HEIGHT + frameThickness / 2, z), windowMetal, shadowGenerator, group)
    createBox(scene, `window-frame-left-${side}`, {
      width: frameThickness,
      height: WINDOW_HEIGHT,
      depth: frameDepth,
    }, new Vector3(-width / 2 + frameThickness / 2, WINDOW_SILL_HEIGHT + WINDOW_HEIGHT / 2, z), windowMetal, shadowGenerator, group)
    createBox(scene, `window-frame-right-${side}`, {
      width: frameThickness,
      height: WINDOW_HEIGHT,
      depth: frameDepth,
    }, new Vector3(width / 2 - frameThickness / 2, WINDOW_SILL_HEIGHT + WINDOW_HEIGHT / 2, z), windowMetal, shadowGenerator, group)
    createBox(scene, `window-mullion-${side}`, {
      width: 0.03,
      height: WINDOW_HEIGHT - 0.02,
      depth: frameDepth * 0.9,
    }, new Vector3(0, WINDOW_SILL_HEIGHT + WINDOW_HEIGHT / 2, z), windowMetal, shadowGenerator, group)

    const glass = MeshBuilder.CreateBox(`window-glass-${side}`, {
      width: width - frameThickness * 2,
      height: WINDOW_HEIGHT - frameThickness * 2,
      depth: paneDepth,
    }, scene)
    glass.position = new Vector3(0, WINDOW_SILL_HEIGHT + WINDOW_HEIGHT / 2, z - side * 0.015)
    glass.material = windowGlass
    glass.parent = group
    glass.receiveShadows = true

    createBox(scene, `window-sill-${side}`, {
      width: width + 0.24,
      height: 0.045,
      depth: 0.18,
    }, new Vector3(0, WINDOW_SILL_HEIGHT - 0.04, z + side * 0.04), windowSill, shadowGenerator, group)
  })

  return group
}

function addFacadeDetails(scene, floorParent, shadowGenerator, materials, floorEnvelope, yBase, floorHeight, options = {}) {
  const width = Math.max(1.2, floorEnvelope.maxX - floorEnvelope.minX)
  const depth = Math.max(1.2, floorEnvelope.maxZ - floorEnvelope.minZ)
  const centerX = (floorEnvelope.minX + floorEnvelope.maxX) / 2
  const centerZ = (floorEnvelope.minZ + floorEnvelope.maxZ) / 2
  const wallMidY = yBase + SLAB_THICKNESS + floorHeight / 2
  const accentMat = materials.facadeAccent || materials.metal || materials.slab
  const buildingMode = Boolean(options.buildingMode)

  createBox(
    scene,
    `facade-front-cornice-${yBase}`,
    { width: width + (buildingMode ? 0.16 : 0.22), height: buildingMode ? 0.05 : 0.07, depth: buildingMode ? 0.06 : 0.08 },
    new Vector3(centerX, yBase + floorHeight + SLAB_THICKNESS - 0.06, floorEnvelope.maxZ + 0.09),
    accentMat,
    shadowGenerator,
    floorParent,
  )

  createBox(
    scene,
    `facade-back-cornice-${yBase}`,
    { width: width + (buildingMode ? 0.16 : 0.26), height: buildingMode ? 0.05 : 0.07, depth: buildingMode ? 0.06 : 0.08 },
    new Vector3(centerX, yBase + floorHeight + SLAB_THICKNESS - 0.06, floorEnvelope.minZ - 0.09),
    accentMat,
    shadowGenerator,
    floorParent,
  )

  if (!buildingMode) {
    createBox(
      scene,
      `facade-left-column-${yBase}`,
      { width: 0.08, height: floorHeight - 0.08, depth: depth + 0.08 },
      new Vector3(floorEnvelope.minX + 0.05, wallMidY, centerZ),
      accentMat,
      shadowGenerator,
      floorParent,
    )

    createBox(
      scene,
      `facade-right-column-${yBase}`,
      { width: 0.08, height: floorHeight - 0.08, depth: depth + 0.08 },
      new Vector3(floorEnvelope.maxX - 0.05, wallMidY, centerZ),
      accentMat,
      shadowGenerator,
      floorParent,
    )
  }

  if (buildingMode) {
    // Building mode stays intentionally restrained for cleaner plan-faithful previews.
  }
}

function createDoor(scene, parent, shadowGenerator, materials, geometryMeta, opening, yBase, thickness, floorHeight, options = {}) {
  const [x1, y1, x2, y2] = opening.hostLine
  const p1 = geometryMeta.transformPoint(x1, y1)
  const p2 = geometryMeta.transformPoint(x2, y2)
  const center = geometryMeta.transformPoint(opening.point.x, opening.point.y)
  const angle = Math.atan2(p2.z - p1.z, p2.x - p1.x)
  const width = Math.max(1.08, opening.widthPx * geometryMeta.scale * 0.48)
  const doorHeight = Math.max(2.1, floorHeight * DOOR_HEIGHT_RATIO)
  const group = new TransformNode("door-group", scene)
  group.parent = parent
  group.position = new Vector3(center.x, yBase, center.z)
  group.rotation = new Vector3(0, angle, 0)
  const facadeOffset = thickness * 0.62
  const trimMaterial = materials.facadeAccent || materials.slab || materials.fallback
  const buildingMode = Boolean(options.buildingMode)

  ;[-1, 1].forEach((side) => {
    const z = facadeOffset * side

    createBox(scene, `door-leaf-${side}`, {
      width,
      height: doorHeight,
      depth: Math.max(0.05, thickness * 0.16),
    }, new Vector3(0, doorHeight / 2, z), materials.wood, shadowGenerator, group)

    createBox(scene, `door-frame-left-${side}`, {
      width: 0.05,
      height: doorHeight + 0.08,
      depth: Math.max(0.08, thickness * 0.42),
    }, new Vector3(-width / 2 - 0.035, doorHeight / 2, z), materials.metal, shadowGenerator, group)
    createBox(scene, `door-frame-right-${side}`, {
      width: 0.05,
      height: doorHeight + 0.08,
      depth: Math.max(0.08, thickness * 0.42),
    }, new Vector3(width / 2 + 0.035, doorHeight / 2, z), materials.metal, shadowGenerator, group)
    createBox(scene, `door-frame-top-${side}`, {
      width: width + 0.12,
      height: 0.08,
      depth: Math.max(0.08, thickness * 0.42),
    }, new Vector3(0, doorHeight + 0.04, z), materials.metal, shadowGenerator, group)

    if (buildingMode) {
      createBox(scene, `door-surround-top-${side}`, {
        width: width + 0.34,
        height: 0.08,
        depth: Math.max(0.09, thickness * 0.34),
      }, new Vector3(0, doorHeight + 0.12, z + side * 0.03), trimMaterial, shadowGenerator, group)
      createBox(scene, `door-surround-left-${side}`, {
        width: 0.07,
        height: doorHeight + 0.22,
        depth: Math.max(0.09, thickness * 0.34),
      }, new Vector3(-width / 2 - 0.14, (doorHeight + 0.1) / 2, z + side * 0.03), trimMaterial, shadowGenerator, group)
      createBox(scene, `door-surround-right-${side}`, {
        width: 0.07,
        height: doorHeight + 0.22,
        depth: Math.max(0.09, thickness * 0.34),
      }, new Vector3(width / 2 + 0.14, (doorHeight + 0.1) / 2, z + side * 0.03), trimMaterial, shadowGenerator, group)
    }
  })

  return group
}

function createRoof(scene, parent, shadowGenerator, materials, envelope, totalHeight) {
  const width = envelope.maxX - envelope.minX
  const depth = envelope.maxZ - envelope.minZ
  createBox(scene, "roof-slab", {
    width: width + 0.12,
    height: Math.max(0.08, ROOF_THICKNESS * 0.72),
    depth: depth + 0.12,
  }, new Vector3((envelope.minX + envelope.maxX) / 2, totalHeight + SLAB_THICKNESS + Math.max(0.08, ROOF_THICKNESS * 0.72) / 2, (envelope.minZ + envelope.maxZ) / 2), materials.roof, shadowGenerator, parent)
}

function createStairCore(scene, parent, shadowGenerator, materials, geometryMeta, stair, yBase, floorHeight) {
  if (!stair?.bounds || stair.bounds.length < 4) return null

  const [x1, y1, x2, y2] = stair.bounds
  const p1 = geometryMeta.transformPoint(x1, y1)
  const p2 = geometryMeta.transformPoint(x2, y2)
  const minX = Math.min(p1.x, p2.x)
  const maxX = Math.max(p1.x, p2.x)
  const minZ = Math.min(p1.z, p2.z)
  const maxZ = Math.max(p1.z, p2.z)
  const width = Math.max(0.4, maxX - minX)
  const depth = Math.max(0.4, maxZ - minZ)
  const steps = Math.max(2, stair.steps || 6)
  const treadDepth = depth / steps
  const riseHeight = Math.min(0.18, Math.max(0.1, floorHeight * 0.055))
  const core = new TransformNode(stair.id || "stair-core", scene)
  core.parent = parent

  for (let index = 0; index < steps; index += 1) {
    const totalDepth = treadDepth * (steps - index)
    const stepHeight = riseHeight * (index + 1)
    const zCenter =
      stair.direction === "down"
        ? maxZ - totalDepth / 2
        : minZ + totalDepth / 2

    createBox(
      scene,
      `${stair.id || "stair"}-step-${index}`,
      {
        width,
        height: stepHeight,
        depth: Math.max(0.08, totalDepth - 0.01),
      },
      new Vector3((minX + maxX) / 2, yBase + stepHeight / 2, zCenter),
      materials.slab || materials.innerTrim || materials.fallback,
      shadowGenerator,
      core,
    )
  }

  return core
}

function addBuildingPolish(scene, parent, shadowGenerator, materials, envelope, floors, floorHeight) {
  if (!envelope || !Number.isFinite(floors) || floors <= 0) return
  if (floors <= 1) return

  const width = envelope.maxX - envelope.minX
  const depth = envelope.maxZ - envelope.minZ
  const centerX = (envelope.minX + envelope.maxX) / 2
  const centerZ = (envelope.minZ + envelope.maxZ) / 2
  const totalBodyHeight = floors * (floorHeight + BUILDING_FLOOR_GAP) - BUILDING_FLOOR_GAP
  const accent = materials.facadeAccent || materials.floorBand || materials.exteriorWall || materials.fallback

  // Floor separation lines remain subtle only for multi-floor reading.
  for (let i = 1; i < floors; i += 1) {
    const y = i * (floorHeight + BUILDING_FLOOR_GAP) + 0.04
    createBox(
      scene,
      `polish-floor-line-front-${i}`,
      { width: width + 0.02, height: 0.025, depth: 0.03 },
      new Vector3(centerX, y, envelope.maxZ + 0.02),
      accent,
      shadowGenerator,
      parent,
    )
    createBox(
      scene,
      `polish-floor-line-back-${i}`,
      { width: width + 0.02, height: 0.025, depth: 0.03 },
      new Vector3(centerX, y, envelope.minZ - 0.02),
      accent,
      shadowGenerator,
      parent,
    )
  }
}

function addTopologyDebugGuides(scene, floorParent, geometryMeta, wallGraph, matched) {
  wallGraph.walls.forEach((wall) => {
    const p1 = geometryMeta.transformPoint(wall.line[0], wall.line[1])
    const p2 = geometryMeta.transformPoint(wall.line[2], wall.line[3])
    const line = MeshBuilder.CreateLines(`dbg-axis-${wall.id}`, {
      points: [new Vector3(p1.x, 0.02, p1.z), new Vector3(p2.x, 0.02, p2.z)],
      updatable: false,
    }, scene)
    line.color = wall.kind === "outer" ? Color3.FromHexString("#4cc9ff") : Color3.FromHexString("#ffce6b")
    line.parent = floorParent
  })

  matched.forEach((opening) => {
    const point = geometryMeta.transformPoint(opening.point.x, opening.point.y)
    const marker = MeshBuilder.CreateSphere(`dbg-opening-${opening.id}`, { diameter: 0.12 }, scene)
    marker.position = new Vector3(point.x, 0.08, point.z)
    const mat = new StandardMaterial(`dbg-opening-mat-${opening.id}`, scene)
    mat.diffuseColor = opening.type === "door"
      ? Color3.FromHexString("#ff6b6b")
      : Color3.FromHexString("#6affaf")
    mat.emissiveColor = mat.diffuseColor.scale(0.45)
    marker.material = mat
    marker.parent = floorParent
  })
}

function logTopologyDebug(wallGraph, matched, floorIndex) {
  console.groupCollapsed(`[Topology] floor=${floorIndex + 1}`)
  wallGraph.walls.forEach((wall) => {
    const wallOpenings = matched
      .filter((opening) => opening.wallId === wall.id)
      .map((opening) => `${opening.type}@t=${opening.t.toFixed(3)} w=${opening.widthPx.toFixed(1)}px`)
      .join(", ")
    console.log(
      `wall=${wall.id} kind=${wall.kind} start=(${wall.line[0].toFixed(1)},${wall.line[1].toFixed(1)}) end=(${wall.line[2].toFixed(1)},${wall.line[3].toFixed(1)}) thickness=${wall.kind === "outer" ? OUTER_WALL_THICKNESS : INNER_WALL_THICKNESS}`,
      wallOpenings || "openings=none"
    )
  })
  console.groupEnd()
}

function createBuildingScene(scene, project, geometryMeta, activeFloor, floorHeight, materials, shadowGenerator, debugOptions = {}) {
  const root = new TransformNode("building-root", scene)
  const floors = project?.floors || []
  console.log("[createBuildingScene] input", {
    floorsLength: floors.length,
    hasGeometryMeta: Boolean(geometryMeta),
    activeFloor,
    mode: activeFloor === "building" ? "building" : "floor",
    firstFloorPolygonLength: floors?.[0]?.polygon?.length || 0,
    firstFloorInnerWallsLength: floors?.[0]?.inner_walls?.length || 0,
    firstFloorDoorsLength: floors?.[0]?.doors?.length || 0,
    firstFloorWindowsLength: floors?.[0]?.windows?.length || 0,
  })
  const envelope = getEnvelope(project, geometryMeta)
  const renderAllFloors = activeFloor === "building"
  const yStep = floorHeight + BUILDING_FLOOR_GAP

  createGround(scene, envelope, materials, shadowGenerator)
  if (renderAllFloors) {
    createPodium(scene, envelope, materials, shadowGenerator, root, {
      includeCap: true,
    })
  }

  floors.forEach((floor, floorIndex) => {
    if (!renderAllFloors && activeFloor !== floorIndex) return

    const floorEnvelope = getFloorEnvelope(floor, geometryMeta)
    const yBase = renderAllFloors ? floorIndex * yStep : 0
    const wallGraph = buildWallGraph({
      polygon: floor.polygon || [],
      innerWalls: floor.inner_walls || [],
    })

    const openings = classifyOpenings({
      graph: wallGraph,
      doors: floor.doors || [],
      windows: floor.windows || [],
      doorWidthPx: DOOR_OPENING_WIDTH_PX,
      windowWidthPx: WINDOW_OPENING_WIDTH_PX,
    })

    const { matched } = matchOpeningsToWalls({
      graph: wallGraph,
      openings,
    })

    const detectedOuterOpenings = matched.filter(
      (opening) => opening.wallKind === "outer" && (opening.type === "window" || opening.type === "door")
    )
    if (renderAllFloors && detectedOuterOpenings.length === 0) {
      console.warn("No reliable facade openings detected for this wall.")
    }

    if (debugOptions?.logTopology) {
      logTopologyDebug(wallGraph, matched, floorIndex)
    }

    const floorParent = new TransformNode(`floor-${floorIndex}`, scene)
    floorParent.parent = root

    const slabInset = 0.04
    const slab = MeshBuilder.CreateBox(`slab-${floorIndex}`, {
      width: Math.max(0.8, floorEnvelope.maxX - floorEnvelope.minX - slabInset),
      height: SLAB_THICKNESS,
      depth: Math.max(0.8, floorEnvelope.maxZ - floorEnvelope.minZ - slabInset),
    }, scene)
    slab.position = new Vector3(
      (floorEnvelope.minX + floorEnvelope.maxX) / 2,
      yBase + SLAB_THICKNESS / 2,
      (floorEnvelope.minZ + floorEnvelope.maxZ) / 2,
    )
    slab.material = materials.slab
    slab.receiveShadows = true
    shadowGenerator.addShadowCaster(slab)
    slab.parent = floorParent

    const openingById = createOpeningLookup(matched)

    const { items } = splitWallsByOpenings({
      graph: wallGraph,
      matchedOpenings: matched,
    })

    items.forEach((item) => {
      if (item.type === "wall") {
        const isOuter = item.wallKind === "outer"
        if (renderAllFloors && isOuter) {
          createBuildingWallSegmentRealistic(
            scene,
            floorParent,
            shadowGenerator,
            materials.exteriorWall,
            materials.innerTrim || materials.innerWall,
            geometryMeta,
            item.line,
            OUTER_WALL_THICKNESS,
            floorHeight,
            yBase + SLAB_THICKNESS,
          )
        } else {
          createWallSegment(
            scene,
            floorParent,
            shadowGenerator,
            isOuter ? materials.exteriorWall : materials.innerWall,
            geometryMeta,
            item.line,
            isOuter ? OUTER_WALL_THICKNESS : INNER_WALL_THICKNESS,
            isOuter ? floorHeight : Math.max(2.58, floorHeight - 0.08),
            yBase + SLAB_THICKNESS,
            {
              isInner: !isOuter,
              trimMaterial: materials.innerTrim || materials.innerWall,
            },
          )
        }
      }

      if (item.type === "door" || item.type === "window") {
        const isOuter = item.wallKind === "outer"
        const wallHeight = isOuter ? floorHeight : Math.max(2.58, floorHeight - 0.08)
        const thickness = isOuter ? OUTER_WALL_THICKNESS : INNER_WALL_THICKNESS
        const sourceOpening = resolveSourceOpening(item, openingById)
        const openingType = sourceOpening?.type || item.type
        const openingWidthWorld = Math.max(0.24, item.widthPx * geometryMeta.scale)
        const wallMaterial = isOuter ? materials.exteriorWall : materials.innerWall

        if (openingType === "door") {
          const doorHeight = Math.max(2.05, wallHeight * DOOR_HEIGHT_RATIO)
          const topFillHeight = wallHeight - doorHeight
          if (topFillHeight > 0.06) {
            createAlignedWallFill(
              scene,
              floorParent,
              shadowGenerator,
              wallMaterial,
              geometryMeta,
              item.hostLine,
              item.point,
              openingWidthWorld,
              yBase + SLAB_THICKNESS + doorHeight + topFillHeight / 2,
              topFillHeight,
              thickness,
            )
          }
          return
        }

        const sillHeight = WINDOW_SILL_HEIGHT
        const openingHeight = Math.min(WINDOW_HEIGHT, wallHeight - sillHeight - 0.12)
        const bottomFillHeight = sillHeight
        const topFillHeight = wallHeight - (sillHeight + openingHeight)

        if (bottomFillHeight > 0.06) {
          createAlignedWallFill(
            scene,
            floorParent,
            shadowGenerator,
            wallMaterial,
            geometryMeta,
            item.hostLine,
            item.point,
            openingWidthWorld,
            yBase + SLAB_THICKNESS + bottomFillHeight / 2,
            bottomFillHeight,
            thickness,
          )
        }

        if (topFillHeight > 0.06) {
          createAlignedWallFill(
            scene,
            floorParent,
            shadowGenerator,
            wallMaterial,
            geometryMeta,
            item.hostLine,
            item.point,
            openingWidthWorld,
            yBase + SLAB_THICKNESS + sillHeight + openingHeight + topFillHeight / 2,
            topFillHeight,
            thickness,
          )
        }
      }
    })

    addFacadeDetails(
      scene,
      floorParent,
      shadowGenerator,
      materials,
      floorEnvelope,
      yBase,
      floorHeight,
      {
        buildingMode: renderAllFloors,
      },
    )

    matched.forEach((opening) => {
      if (opening.type === "window" && opening.wallKind === "outer") {
        createWindow(
          scene,
          floorParent,
          shadowGenerator,
          materials,
          geometryMeta,
          opening,
          yBase + SLAB_THICKNESS,
          OUTER_WALL_THICKNESS,
          { buildingMode: renderAllFloors },
        )
      } else if (opening.type === "door") {
        createDoor(
          scene,
          floorParent,
          shadowGenerator,
          materials,
          geometryMeta,
          opening,
          yBase + SLAB_THICKNESS,
          opening.wallKind === "outer" ? OUTER_WALL_THICKNESS : INNER_WALL_THICKNESS,
          floorHeight,
          { buildingMode: renderAllFloors },
        )
      }
    })

    ;(floor.stairs || []).forEach((stair) => {
      createStairCore(
        scene,
        floorParent,
        shadowGenerator,
        materials,
        geometryMeta,
        stair,
        yBase + SLAB_THICKNESS,
        floorHeight,
      )
    })

    if (debugOptions?.enabled) {
      addTopologyDebugGuides(scene, floorParent, geometryMeta, wallGraph, matched)
    }

    if (renderAllFloors && floorIndex < floors.length - 1) {
      const centerX = (floorEnvelope.minX + floorEnvelope.maxX) / 2
      const centerZ = (floorEnvelope.minZ + floorEnvelope.maxZ) / 2
      const width = Math.max(1.2, floorEnvelope.maxX - floorEnvelope.minX)
      const depth = Math.max(1.2, floorEnvelope.maxZ - floorEnvelope.minZ)
      createBox(
        scene,
        `floor-band-${floorIndex}`,
        { width, height: 0.06, depth },
        new Vector3(centerX, yBase + floorHeight + SLAB_THICKNESS + 0.01, centerZ),
        materials.floorBand || materials.slab,
        shadowGenerator,
        floorParent,
      )
    }

    if (renderAllFloors && floorIndex === floors.length - 1) {
      const centerX = (floorEnvelope.minX + floorEnvelope.maxX) / 2
      const centerZ = (floorEnvelope.minZ + floorEnvelope.maxZ) / 2
      const topClosureInset = 0.08
      const topClosure = MeshBuilder.CreateBox(`top-closure-${floorIndex}`, {
        width: Math.max(0.8, floorEnvelope.maxX - floorEnvelope.minX - topClosureInset),
        height: Math.max(0.05, SLAB_THICKNESS * 0.5),
        depth: Math.max(0.8, floorEnvelope.maxZ - floorEnvelope.minZ - topClosureInset),
      }, scene)
      topClosure.position = new Vector3(
        centerX,
        yBase + SLAB_THICKNESS + floorHeight - Math.max(0.04, SLAB_THICKNESS * 0.36),
        centerZ,
      )
      topClosure.material = materials.slab
      topClosure.receiveShadows = true
      shadowGenerator.addShadowCaster(topClosure)
      topClosure.parent = floorParent
    }
  })

  if (renderAllFloors) {
    createRoof(scene, root, shadowGenerator, materials, envelope, floors.length * yStep - BUILDING_FLOOR_GAP)
    addBuildingPolish(
      scene,
      root,
      shadowGenerator,
      materials,
      envelope,
      floors.length,
      floorHeight,
    )
  }

  return root
}

function fitCameraToEnvelope(camera, envelope, activeFloor, totalHeight) {
  const width = envelope.maxX - envelope.minX
  const depth = envelope.maxZ - envelope.minZ
  const maxDimension = Math.max(width, depth, totalHeight)
  const center = new Vector3(
    (envelope.minX + envelope.maxX) / 2,
    activeFloor === "building" ? totalHeight * 0.42 : totalHeight * 0.34,
    (envelope.minZ + envelope.maxZ) / 2,
  )

  camera.setTarget(center)
  camera.radius = Math.max(8.1, maxDimension * (activeFloor === "building" ? 1.02 : 1.12))
  camera.alpha = activeFloor === "building" ? -Math.PI / 4.15 : -Math.PI / 3.6
  camera.beta = activeFloor === "building" ? 0.98 : 1.02
}

function addDebugMarker(scene, target) {
  const marker = MeshBuilder.CreateBox("debug-marker", {
    size: 0.8,
  }, scene)
  const material = new StandardMaterial("debug-marker-mat", scene)
  material.diffuseColor = Color3.FromHexString("#ff5a3d")
  material.emissiveColor = Color3.FromHexString("#ff5a3d")
  marker.material = material
  marker.position = target.clone()
  marker.position.y += 1
  return marker
}

function getEnvelope(project, geometryMeta) {
  const allPoints = []
  ;(project?.floors || []).forEach((floor) => {
    ;(floor.polygon || []).forEach(([x, y]) => allPoints.push(geometryMeta.transformPoint(x, y)))
  })

  return {
    minX: Math.min(...allPoints.map((point) => point.x)),
    maxX: Math.max(...allPoints.map((point) => point.x)),
    minZ: Math.min(...allPoints.map((point) => point.z)),
    maxZ: Math.max(...allPoints.map((point) => point.z)),
  }
}

function getFloorEnvelope(floor, geometryMeta) {
  const points = (floor?.polygon || []).map(([x, y]) => geometryMeta.transformPoint(x, y))
  return {
    minX: Math.min(...points.map((point) => point.x)),
    maxX: Math.max(...points.map((point) => point.x)),
    minZ: Math.min(...points.map((point) => point.z)),
    maxZ: Math.max(...points.map((point) => point.z)),
  }
}

function createFacadeMass(scene, parent, shadowGenerator, materials, floorEnvelope, yBase, floorHeight) {
  const width = Math.max(1.2, floorEnvelope.maxX - floorEnvelope.minX)
  const depth = Math.max(1.2, floorEnvelope.maxZ - floorEnvelope.minZ)
  const centerX = (floorEnvelope.minX + floorEnvelope.maxX) / 2
  const centerZ = (floorEnvelope.minZ + floorEnvelope.maxZ) / 2
  const wallHeight = Math.max(2.6, floorHeight)

  createBox(
    scene,
    "facade-front",
    {
      width,
      height: wallHeight,
      depth: OUTER_WALL_THICKNESS,
    },
    new Vector3(centerX, yBase + SLAB_THICKNESS + wallHeight / 2, floorEnvelope.maxZ),
    materials.exteriorWall,
    shadowGenerator,
    parent,
  )

  createBox(
    scene,
    "facade-back",
    {
      width,
      height: wallHeight,
      depth: OUTER_WALL_THICKNESS,
    },
    new Vector3(centerX, yBase + SLAB_THICKNESS + wallHeight / 2, floorEnvelope.minZ),
    materials.exteriorWall,
    shadowGenerator,
    parent,
  )

  createBox(
    scene,
    "facade-left",
    {
      width: OUTER_WALL_THICKNESS,
      height: wallHeight,
      depth,
    },
    new Vector3(floorEnvelope.minX, yBase + SLAB_THICKNESS + wallHeight / 2, centerZ),
    materials.exteriorWall,
    shadowGenerator,
    parent,
  )

  createBox(
    scene,
    "facade-right",
    {
      width: OUTER_WALL_THICKNESS,
      height: wallHeight,
      depth,
    },
    new Vector3(floorEnvelope.maxX, yBase + SLAB_THICKNESS + wallHeight / 2, centerZ),
    materials.exteriorWall,
    shadowGenerator,
    parent,
  )

  createBox(
    scene,
    "floor-plinth-band",
    {
      width: width + 0.18,
      height: 0.22,
      depth: depth + 0.18,
    },
    new Vector3(centerX, yBase + 0.11, centerZ),
    materials.plinth,
    shadowGenerator,
    parent,
  )
}

function createDebugWallSegment(scene, parent, geometryMeta, line, material, yBase, height, thickness, shadowGenerator) {
  const [x1, y1, x2, y2] = line
  const p1 = geometryMeta.transformPoint(x1, y1)
  const p2 = geometryMeta.transformPoint(x2, y2)
  const dx = p2.x - p1.x
  const dz = p2.z - p1.z
  const length = Math.hypot(dx, dz)

  if (length < 0.08) return null

  const mesh = MeshBuilder.CreateBox(`debug-wall-${x1}-${y1}-${x2}-${y2}`, {
    width: length,
    height,
    depth: thickness,
  }, scene)
  mesh.position = new Vector3((p1.x + p2.x) / 2, yBase + height / 2, (p1.z + p2.z) / 2)
  mesh.rotation.y = Math.atan2(dz, dx)
  mesh.material = material
  mesh.receiveShadows = true
  shadowGenerator?.addShadowCaster(mesh)
  mesh.parent = parent
  return mesh
}

function createMinimalBuildingScene(scene, project, geometryMeta, activeFloor, floorHeight, materials, shadowGenerator) {
  const root = new TransformNode("debug-building-root", scene)
  const floors = project?.floors || []
  const renderAllFloors = activeFloor === "building"
  const targetFloors = renderAllFloors ? floors : [floors[activeFloor]].filter(Boolean)
  const yStep = floorHeight + BUILDING_FLOOR_GAP
  const wallMaterial = materials.exteriorWall || materials.fallback
  const roofMaterial = materials.roof || materials.fallback
  const floorMaterial = materials.slab || materials.fallback

  const buildingEnvelope = getEnvelope(project, geometryMeta)
  createGround(scene, buildingEnvelope, materials, shadowGenerator)
  if (renderAllFloors) {
    createPodium(scene, buildingEnvelope, materials, shadowGenerator, root, {
      includeCap: true,
    })
  }
  // Building view stays clean for now: no road/forecourt context meshes.

  console.log("[Babylon debug] floors:", floors.length, "activeFloor:", activeFloor)

  let createdMeshes = 0

  targetFloors.forEach((floor, localIndex) => {
    const sourceIndex = renderAllFloors ? localIndex : activeFloor
    const polygon = floor?.polygon || []
    console.log("[Babylon debug] floor polygon", sourceIndex, polygon)

    if (!polygon.length) return

    const wallGraph = buildWallGraph({
      polygon: floor.polygon || [],
      innerWalls: floor.inner_walls || [],
    })

    const openings = classifyOpenings({
      graph: wallGraph,
      doors: floor.doors || [],
      windows: floor.windows || [],
      doorWidthPx: DOOR_OPENING_WIDTH_PX,
      windowWidthPx: WINDOW_OPENING_WIDTH_PX,
    })

    const { matched } = matchOpeningsToWalls({
      graph: wallGraph,
      openings,
    })

    const floorEnvelope = getFloorEnvelope(floor, geometryMeta)
    const width = Math.max(2, floorEnvelope.maxX - floorEnvelope.minX)
    const depth = Math.max(2, floorEnvelope.maxZ - floorEnvelope.minZ)
    const yBase = renderAllFloors ? sourceIndex * yStep : 0
    const centerX = (floorEnvelope.minX + floorEnvelope.maxX) / 2
    const centerZ = (floorEnvelope.minZ + floorEnvelope.maxZ) / 2
    const wallHeight = Math.max(2.8, floorHeight)
    const floorParent = new TransformNode(`debug-floor-${sourceIndex}`, scene)
    floorParent.parent = root

    createBox(
      scene,
      `plinth-band-${sourceIndex}`,
      {
        width: width + 0.28,
        height: 0.22,
        depth: depth + 0.28,
      },
      new Vector3(centerX, yBase + 0.11, centerZ),
      materials.plinth || materials.fallback,
      shadowGenerator,
      floorParent,
    )

    createBox(
      scene,
      `belt-line-front-${sourceIndex}`,
      {
        width: width + 0.12,
        height: 0.05,
        depth: 0.12,
      },
      new Vector3(centerX, yBase + floorHeight - 0.02, floorEnvelope.maxZ + 0.03),
      materials.metal || materials.fallback,
      shadowGenerator,
      floorParent,
    )

    createBox(
      scene,
      `belt-line-back-${sourceIndex}`,
      {
        width: width + 0.12,
        height: 0.05,
        depth: 0.12,
      },
      new Vector3(centerX, yBase + floorHeight - 0.02, floorEnvelope.minZ - 0.03),
      materials.metal || materials.fallback,
      shadowGenerator,
      floorParent,
    )

    const openingById = createOpeningLookup(matched)

    const { items } = splitWallsByOpenings({
      graph: wallGraph,
      matchedOpenings: matched,
    })

    items.forEach((item) => {
      if (item.type === "wall") {
        const isOuter = item.wallKind === "outer"
        const wall = createDebugWallSegment(
          scene,
          floorParent,
          geometryMeta,
          item.line,
          isOuter ? wallMaterial : (materials.innerWall || wallMaterial),
          yBase + SLAB_THICKNESS,
          isOuter ? wallHeight : Math.max(2.58, wallHeight - 0.08),
          isOuter ? OUTER_WALL_THICKNESS : INNER_WALL_THICKNESS,
          shadowGenerator,
        )
        if (wall) createdMeshes += 1
      }

      if (item.type === "door" || item.type === "window") {
        const isOuter = item.wallKind === "outer"
        const activeWallHeight = isOuter ? wallHeight : Math.max(2.58, wallHeight - 0.08)
        const thickness = isOuter ? OUTER_WALL_THICKNESS : INNER_WALL_THICKNESS
        const sourceOpening = resolveSourceOpening(item, openingById)
        const openingType = sourceOpening?.type || item.type
        const openingWidthWorld = Math.max(0.24, item.widthPx * geometryMeta.scale)
        const wallMat = isOuter ? wallMaterial : (materials.innerWall || wallMaterial)

        if (openingType === "door") {
          const doorHeight = Math.max(2.05, activeWallHeight * DOOR_HEIGHT_RATIO)
          const topFillHeight = activeWallHeight - doorHeight
          if (topFillHeight > 0.06) {
            const fill = createAlignedWallFill(
              scene,
              floorParent,
              shadowGenerator,
              wallMat,
              geometryMeta,
              item.hostLine,
              item.point,
              openingWidthWorld,
              yBase + SLAB_THICKNESS + doorHeight + topFillHeight / 2,
              topFillHeight,
              thickness,
            )
            if (fill) createdMeshes += 1
          }
          return
        }

        const sillHeight = WINDOW_SILL_HEIGHT
        const openingHeight = Math.min(WINDOW_HEIGHT, activeWallHeight - sillHeight - 0.12)
        const bottomFillHeight = sillHeight
        const topFillHeight = activeWallHeight - (sillHeight + openingHeight)

        if (bottomFillHeight > 0.06) {
          const fill = createAlignedWallFill(
            scene,
            floorParent,
            shadowGenerator,
            wallMat,
            geometryMeta,
            item.hostLine,
            item.point,
            openingWidthWorld,
            yBase + SLAB_THICKNESS + bottomFillHeight / 2,
            bottomFillHeight,
            thickness,
          )
          if (fill) createdMeshes += 1
        }
        if (topFillHeight > 0.06) {
          const fill = createAlignedWallFill(
            scene,
            floorParent,
            shadowGenerator,
            wallMat,
            geometryMeta,
            item.hostLine,
            item.point,
            openingWidthWorld,
            yBase + SLAB_THICKNESS + sillHeight + openingHeight + topFillHeight / 2,
            topFillHeight,
            thickness,
          )
          if (fill) createdMeshes += 1
        }
      }
    })

    const floorPlate = MeshBuilder.CreateBox(`debug-floor-plate-${sourceIndex}`, {
      width: width + 0.04,
      height: 0.08,
      depth: depth + 0.04,
    }, scene)
    floorPlate.position = new Vector3(centerX, yBase + 0.04, centerZ)
    floorPlate.material = floorMaterial
    floorPlate.receiveShadows = true
    shadowGenerator?.addShadowCaster(floorPlate)
    floorPlate.parent = floorParent
    createdMeshes += 1

    const top = MeshBuilder.CreateBox(`debug-floor-top-${sourceIndex}`, {
      width: width + 0.08,
      height: 0.12,
      depth: depth + 0.08,
    }, scene)
    top.position = new Vector3(centerX, yBase + wallHeight + 0.06, centerZ)
    top.material = renderAllFloors && sourceIndex === floors.length - 1 ? roofMaterial : floorMaterial
    top.receiveShadows = true
    shadowGenerator?.addShadowCaster(top)
    top.parent = floorParent
    createdMeshes += 1

    matched.forEach((opening) => {
      if (opening.type === "window" && opening.wallKind === "outer") {
        createWindow(
          scene,
          floorParent,
          shadowGenerator,
          materials,
          geometryMeta,
          opening,
          yBase + SLAB_THICKNESS,
          OUTER_WALL_THICKNESS,
          { buildingMode: renderAllFloors },
        )
        createdMeshes += 1
      } else if (opening.type === "door" && sourceIndex === 0) {
        createDoor(
          scene,
          floorParent,
          shadowGenerator,
          materials,
          geometryMeta,
          opening,
          yBase + SLAB_THICKNESS,
          opening.wallKind === "outer" ? OUTER_WALL_THICKNESS : INNER_WALL_THICKNESS,
          floorHeight,
          { buildingMode: renderAllFloors },
        )
        createdMeshes += 1
      }
    })

    if (sourceIndex === 0) {
      createBox(
        scene,
        "entry-accent",
        {
          width: Math.max(1.5, width * 0.16),
          height: floorHeight + 0.5,
          depth: 0.14,
        },
        new Vector3(centerX, yBase + (floorHeight + 0.5) / 2, floorEnvelope.maxZ + 0.08),
        materials.wood || materials.fallback,
        shadowGenerator,
        floorParent,
      )
    }
  })

  if (createdMeshes === 0) {
    const fallback = MeshBuilder.CreateBox("debug-fallback-box", {
      width: 10,
      height: 5,
      depth: 10,
    }, scene)
    fallback.position = Vector3.Zero()
    fallback.material = materials.fallback
    fallback.receiveShadows = true
    shadowGenerator?.addShadowCaster(fallback)
    fallback.parent = root
    createdMeshes = 1
  }

  if (renderAllFloors) {
    createRoof(
      scene,
      root,
      shadowGenerator,
      materials,
      buildingEnvelope,
      floors.length * yStep - BUILDING_FLOOR_GAP,
    )
    createdMeshes += 1
  }

  console.log("[Babylon debug] mesh count:", createdMeshes)
  return root
}

export function createBabylonViewer({
  canvas,
  project,
  geometryMeta,
  activeFloor,
  floorHeight,
  visualOverrides = {},
  debugOptions = {},
}) {
  const engine = new Engine(canvas, true, {
    preserveDrawingBuffer: true,
    stencil: true,
    disableWebGL2Support: false,
  })
  const scene = new Scene(engine)
  scene.clearColor = new Color4(0.92, 0.94, 0.96, 1)

  const camera = new ArcRotateCamera("camera", -Math.PI / 4, 1.05, 20, Vector3.Zero(), scene)
  scene.activeCamera = camera
  camera.setTarget(Vector3.Zero())
  camera.radius = 20
  camera.lowerRadiusLimit = 6
  camera.upperRadiusLimit = 80
  camera.lowerBetaLimit = 0.55
  camera.upperBetaLimit = 1.42
  camera.panningSensibility = 0
  camera.attachControl(canvas, true)
  engine.resize()

  const isBuildingView = activeFloor === "building"
  const environmentTexture = setupEnvironment(scene, isBuildingView)
  scene.imageProcessingConfiguration.toneMappingEnabled = true
  scene.imageProcessingConfiguration.toneMappingType = ImageProcessingConfiguration.TONEMAPPING_ACES
  scene.imageProcessingConfiguration.exposure = isBuildingView ? 1.12 : 1.08
  scene.imageProcessingConfiguration.contrast = isBuildingView ? 1.12 : 1.16
  scene.imageProcessingConfiguration.vignetteEnabled = false
  scene.fogMode = Scene.FOGMODE_NONE

  const { shadowGenerator, disposeLighting } = setupLighting(scene, activeFloor, camera)
  const baseMaterials = createMaterialLibrary(scene, environmentTexture)
  const materials = isBuildingView
    ? createBuildingMaterialVariant(scene, baseMaterials, environmentTexture)
    : baseMaterials
  applyVisualOverrides(materials, visualOverrides)
  const disposePostFx = isBuildingView ? setupBuildingPostFx(scene, camera) : () => {}
  let root = null

  root = createBuildingScene(
    scene,
    project,
    geometryMeta,
    activeFloor,
    floorHeight,
    materials,
    shadowGenerator,
    debugOptions,
  )
  if (activeFloor !== "building") {
    const floorEnvelope = getFloorEnvelope(project?.floors?.[activeFloor], geometryMeta)
    fitCameraToEnvelope(camera, floorEnvelope, activeFloor, floorHeight + 1.2)
  } else {
    const env = getEnvelope(project, geometryMeta)
    fitCameraToEnvelope(
      camera,
      env,
      "building",
      (project?.floors?.length || 1) * (floorHeight + BUILDING_FLOOR_GAP)
    )
  }

  const resize = () => engine.resize()
  window.addEventListener("resize", resize)

  engine.runRenderLoop(() => {
    scene.render()
  })

  const exportNodeIds = new Set(
    [root, ...(root?.getDescendants?.(false) || [])]
      .filter(Boolean)
      .map((node) => node.uniqueId)
  )

  const exportOptions = {
    shouldExportNode: (node) => exportNodeIds.has(node.uniqueId),
  }

  return {
    engine,
    scene,
    camera,
    root,
    dispose() {
      window.removeEventListener("resize", resize)
      disposePostFx?.()
      disposeLighting?.()
      scene.dispose()
      engine.dispose()
    },
    async exportGlb(filenameBase) {
      const glb = await GLTF2Export.GLBAsync(scene, filenameBase, exportOptions)
      glb.downloadFiles()
    },
    async exportGlbBlob(filenameBase) {
      return exportProjectGlb(project, geometryMeta, floorHeight, filenameBase)
    },
    screenshot() {
      return canvas.toDataURL("image/png")
    },
  }
}

export function disposeBabylonViewer(viewer) {
  if (viewer) viewer.dispose()
}

export function downloadBabylonScreenshot(viewer, filename) {
  if (!viewer) return
  const url = viewer.screenshot()
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}


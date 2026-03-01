import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
function App() {
  const [data, setData] = useState(null);
  const [height, setHeight] = useState(8);
  const canvasRef = useRef(null);
  const threeCanvasRef = useRef(null);
  const threeRef = useRef({
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  mesh: null,
  animId: null,
});

 // 1️⃣ Three.js sahne kurulumu
useEffect(() => {
  if (!data) return;

  const canvas = threeCanvasRef.current;
  if (!canvas) return;

  const W = 500;
  const H = 350;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(W, H);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111111);

  const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 1000);
  camera.position.set(15, 15, 15);

  const ambient = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambient);

  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(10, 20, 10);
  scene.add(dir);

  const grid = new THREE.GridHelper(50, 50);
  scene.add(grid);
  if (data && data.polygon) {

  const shape = new THREE.Shape();

  const pts = data.polygon;

  shape.moveTo(pts[0][0], pts[0][1]);

  for (let i = 1; i < pts.length; i++) {
    shape.lineTo(pts[i][0], pts[i][1]);
  }

  const extrudeSettings = {
    depth: data.height,
    bevelEnabled: false,
  };

  const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);

  const material = new THREE.MeshStandardMaterial({
    color: 0x3fa9f5,
    metalness: 0.2,
    roughness: 0.7,
  });

  const mesh = new THREE.Mesh(geometry, material);

  scene.add(mesh);
}

  const controls = new OrbitControls(camera, renderer.domElement);

  const animate = () => {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  };
  animate();

}, [data]);


// 2️⃣ Fetch effect
useEffect(() => {
  console.log("fetch effect çalıştı");
  fetch(`http://127.0.0.1:8000/demo/polygon?height=${height}`)
    .then((response) => response.json())
    .then((result) => setData(result))
    .catch((error) => console.error("Error fetching data:", error));
}, [height]);
  useEffect(() => {
    if (!canvasRef.current) return;
    if (!data || !data.polygon || data.polygon.length < 2) return;

    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;

    const W = 500,
      H = 350;

    // Transformları sıfırla
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // Temizle
    ctx.clearRect(0, 0, W, H);

    // Y eksenini ters çevir (0,0 sol-alt olsun)
    ctx.translate(0, H);
    ctx.scale(1, -1);

    const pts = data.polygon;

    // demo: ölçek + kaydırma
    const scale = 20;
    const offsetX = 80; 
    const offsetY = 80; // istersen kullan, şimdilik çizimi yukarı taşır

    ctx.beginPath();
    ctx.moveTo(offsetX + pts[0][0] * scale, offsetY + pts[0][1] * scale);

    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(offsetX + pts[i][0] * scale, offsetY + pts[i][1] * scale);
    }

    ctx.closePath();

    ctx.strokeStyle = "red";
    ctx.lineWidth = 3;
    ctx.stroke();
    const dx = 30;               // sağa kaydırma (3D derinlik)
const dy = 20;               // yukarı kaydırma (3D derinlik)
const extrudeScale = 2;      // height'i ne kadar yansıtalım
const z = (data.height ?? 0) * extrudeScale;
// Üst yüzey (polygonu z kadar offset'li çiz)
ctx.beginPath();
ctx.moveTo(
  offsetX + pts[0][0] * scale + dx,
  offsetY + pts[0][1] * scale + dy + z
);

for (let i = 1; i < pts.length; i++) {
  ctx.lineTo(
    offsetX + pts[i][0] * scale + dx,
    offsetY + pts[i][1] * scale + dy + z
  );
}
ctx.closePath();
ctx.stroke();
// Yan yüzey bağlantı çizgileri
for (let i = 0; i < pts.length; i++) {
  const j = (i + 1) % pts.length;

  const x1 = offsetX + pts[i][0] * scale;
  const y1 = offsetY + pts[i][1] * scale;

  const x1Top = x1 + dx;
  const y1Top = y1 + dy + z;

  // alt köşeden üst köşeye dik çizgi
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1Top, y1Top);
  ctx.stroke();
}
  }, [data]);

 return (
  <div style={{ padding: "40px", fontFamily: "Arial" }}>
    <h1>Plan2Mass Demo</h1>
<div style={{ marginBottom: "20px" }}>
  <label style={{ display: "block", marginBottom: "8px" }}>
    Height: <b>{height}</b>
  </label>

  <input
    type="range"
    min="1"
    max="50"
    value={height}
    onChange={(e) => setHeight(Number(e.target.value))}
    style={{ width: "500px" }}
  />
</div>
    {data ? (
      <div>
        <h3>Height (state): {height} | (api): {data.height}</h3>
        <h3>Polygon:</h3>
        <pre>{JSON.stringify(data.polygon, null, 2)}</pre>

        {/* 2D Canvas */}
        <canvas
          ref={canvasRef}
          width="500"
          height="350"
          style={{ border: "1px solid #555", marginTop: "20px" }}
        ></canvas>

        {/* 3D Alan */}
        <canvas
          ref={threeCanvasRef}
          style={{
            width: "500px",
            height: "350px",
            marginTop: "30px",
            border: "1px solid #888"
          }}
        ></canvas>
      </div>
    ) : (
      <p>Loading...</p>
    )}
  </div>
);
}

export default App;

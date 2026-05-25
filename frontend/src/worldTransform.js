export function buildWorldTransform({
  minX,
  maxX,
  minY,
  maxY,
  targetSize = 22,
}) {
  const width = Math.max(1, maxX - minX)
  const depth = Math.max(1, maxY - minY)
  const maxDim = Math.max(width, depth)
  const scale = targetSize / maxDim
  const centerX = (minX + maxX) / 2
  const centerY = (minY + maxY) / 2

  const worldTransform = (x, y) => ({
    x: (x - centerX) * scale,
    z: -(y - centerY) * scale,
  })

  return {
    width,
    depth,
    maxDim,
    scale,
    centerX,
    centerY,
    worldTransform,
    transformPoint: worldTransform,
  }
}

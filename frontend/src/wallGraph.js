const DEFAULTS = {
  endpointTolerancePx: 12,
  duplicateWallTolerancePx: 6,
  minSegmentLengthPx: 18,
  minInnerSegmentLengthPx: 24,
  projectionMargin: 0.04,
  minWallPiecePx: 16,
  minEdgePiecePx: 14,
  overlapJoinGapPx: 3,
  parallelMergeDistancePx: 6,
  collinearGapPx: 8,
  axisSkewRatio: 1.35,
  splitIntersectionTolerancePx: 3,
  outerDoorDistancePx: 28,
  innerDoorDistancePx: 22,
  outerWindowDistancePx: 30,
}

export function buildWallGraph({
  polygon = [],
  innerWalls = [],
  options = {},
}) {
  const cfg = { ...DEFAULTS, ...options }
  const issues = []

  const mergedInnerWalls = mergeNearParallelInnerWalls(innerWalls, cfg)

  const rawWalls = [
    ...polygonToSegments(polygon).map((line, index) => ({
      id: `outer-${index}`,
      line,
      kind: "outer",
    })),
    ...mergedInnerWalls.map((line, index) => ({
      id: `inner-${index}`,
      line,
      kind: "inner",
    })),
  ]

  const normalizedWalls = splitWallsAtIntersections(
    dedupeWalls(rawWalls, cfg),
    cfg
  )

  const walls = normalizedWalls.map((item) =>
    createWallSegment(item.id, item.line, item.kind, cfg.minSegmentLengthPx)
  ).filter(Boolean)

  const endpointBuckets = new Map()

  walls.forEach((wall) => {
    for (const endpoint of [wall.start, wall.end]) {
      const key = bucketKey(endpoint.x, endpoint.y, cfg.endpointTolerancePx)
      if (!endpointBuckets.has(key)) endpointBuckets.set(key, [])
      endpointBuckets.get(key).push(wall.id)
    }
  })

  const wallById = new Map(walls.map((wall) => [wall.id, wall]))

  walls.forEach((wall) => {
    const adjacent = new Set()

    for (const endpoint of [wall.start, wall.end]) {
      const keys = neighboringBucketKeys(endpoint.x, endpoint.y, cfg.endpointTolerancePx)
      keys.forEach((key) => {
        for (const neighborId of endpointBuckets.get(key) || []) {
          if (neighborId !== wall.id) adjacent.add(neighborId)
        }
      })
    }

    wall.adjacentWallIds = [...adjacent]

    if (wall.length < cfg.minSegmentLengthPx * 1.5) {
      issues.push({
        type: "short-wall",
        wallId: wall.id,
        line: wall.line,
      })
    }
  })

  return {
    walls,
    wallById,
    options: cfg,
    issues,
  }
}

export function classifyOpenings({
  graph,
  doors = [],
  windows = [],
  doorWidthPx,
  windowWidthPx,
  options = {},
}) {
  const cfg = { ...graph.options, ...options }
  const classified = []

  doors.forEach((point, index) => {
    const opening = makeOpening(`door-${index}`, "door", point, doorWidthPx)

    const innerBest = findBestCandidate(opening, graph.walls, {
      allowedKinds: ["inner"],
      maxDistancePx: cfg.innerDoorDistancePx,
      projectionMargin: cfg.projectionMargin,
    })

    const outerBest = findBestCandidate(opening, graph.walls, {
      allowedKinds: ["outer"],
      maxDistancePx: cfg.outerDoorDistancePx,
      projectionMargin: cfg.projectionMargin,
    })

    const rejectedCandidates = [
      ...(innerBest?.debug?.rejectedCandidates || []),
      ...(outerBest?.debug?.rejectedCandidates || []),
    ]
    const candidateScores = [innerBest, outerBest]
      .filter(Boolean)
      .map((candidate) => ({
        wallId: candidate.wall.id,
        wallKind: candidate.wall.kind,
        score: roundDebugValue(candidate.score),
        distancePx: roundDebugValue(candidate.distancePx),
      }))
      .sort((a, b) => a.score - b.score)

    if (candidateScores.length >= 2 && Math.abs(candidateScores[0].score - candidateScores[1].score) <= 1.5) {
      console.warn("AMBIGUOUS_OPENING_MATCH", {
        openingId: opening.id,
        type: opening.type,
        centerPoint: opening.point,
        widthPx: opening.widthPx,
        candidateScores,
      })
    }

    if (innerBest) {
      console.log("[classifyOpenings] opening classified", {
        openingId: opening.id,
        type: opening.type,
        centerPoint: opening.point,
        widthPx: opening.widthPx,
        chosenHostWall: innerBest.wall.id,
        hostWallType: innerBest.wall.kind,
        score: roundDebugValue(innerBest.score),
        rejectedCandidates,
      })
      classified.push({
        ...opening,
        preferredKinds: ["inner"],
        preferredWallId: innerBest.wall.id,
        preferredProjection: innerBest.projected,
        previewMatch: innerBest,
      })
      return
    }

    if (outerBest) {
      console.log("[classifyOpenings] opening classified", {
        openingId: opening.id,
        type: opening.type,
        centerPoint: opening.point,
        widthPx: opening.widthPx,
        chosenHostWall: outerBest.wall.id,
        hostWallType: outerBest.wall.kind,
        score: roundDebugValue(outerBest.score),
        rejectedCandidates,
      })
      classified.push({
        ...opening,
        preferredKinds: ["outer"],
        preferredWallId: outerBest.wall.id,
        preferredProjection: outerBest.projected,
        previewMatch: outerBest,
      })
      return
    }

    console.warn("OPENING_MATCH_FAILED", {
      phase: "classifyOpenings",
      openingId: opening.id,
      type: opening.type,
      centerPoint: opening.point,
      widthPx: opening.widthPx,
      rejectedCandidates,
    })
  })

  windows.forEach((point, index) => {
    const opening = makeOpening(`window-${index}`, "window", point, windowWidthPx)
    const allowedKinds = opening.preferredKinds?.length ? opening.preferredKinds : ["outer"]
    const best = findBestCandidate(opening, graph.walls, {
      allowedKinds,
      maxDistancePx: allowedKinds.includes("inner") && !allowedKinds.includes("outer")
        ? cfg.innerDoorDistancePx
        : cfg.outerWindowDistancePx,
      projectionMargin: cfg.projectionMargin,
    })

    if (best) {
      console.log("[classifyOpenings] opening classified", {
        openingId: opening.id,
        type: opening.type,
        centerPoint: opening.point,
        widthPx: opening.widthPx,
        chosenHostWall: best.wall.id,
        hostWallType: best.wall.kind,
        score: roundDebugValue(best.score),
        rejectedCandidates: best?.debug?.rejectedCandidates || [],
      })
      classified.push({
        ...opening,
        preferredKinds: allowedKinds,
        preferredWallId: opening.preferredWallId || best.wall.id,
        preferredProjection: best.projected,
        previewMatch: best,
      })
      return
    }

    console.warn("OPENING_MATCH_FAILED", {
      phase: "classifyOpenings",
      openingId: opening.id,
      type: opening.type,
      centerPoint: opening.point,
      widthPx: opening.widthPx,
      preferredKinds: allowedKinds,
    })
  })

  return classified
}

export function matchOpeningsToWalls({
  graph,
  openings = [],
  options = {},
}) {
  const cfg = { ...graph.options, ...options }
  const matched = []
  const unmatched = []

  openings.forEach((opening) => {
    const maxDistancePx =
      opening.type === "window"
        ? cfg.outerWindowDistancePx
        : opening.preferredKinds?.[0] === "inner"
          ? cfg.innerDoorDistancePx
          : cfg.outerDoorDistancePx

    if (opening.preferredWallId) {
      const preferredWall = graph.wallById.get(opening.preferredWallId)
      if (preferredWall) {
        const direct = findBestCandidate(opening, [preferredWall], {
          allowedKinds: [preferredWall.kind],
          maxDistancePx: maxDistancePx * 1.35,
          projectionMargin: 0,
        })
        if (direct) {
          console.log("[matchOpeningsToWalls] opening matched via preferred wall", {
            openingId: opening.id,
            type: opening.type,
            centerPoint: opening.point,
            widthPx: opening.widthPx,
            chosenHostWall: direct.wall.id,
            hostWallType: direct.wall.kind,
            score: roundDebugValue(direct.score),
            rejectedCandidates: direct?.debug?.rejectedCandidates || [],
          })
          matched.push({
            ...opening,
            wallId: direct.wall.id,
            wallKind: direct.wall.kind,
            hostLine: [...direct.wall.line],
            point: {
              x: direct.projected.x,
              y: direct.projected.y,
            },
            t: direct.projected.t,
            distancePx: direct.distancePx,
            score: direct.score,
          })
          return
        }
      }
    }

    const best = findBestCandidate(opening, graph.walls, {
      allowedKinds: opening.preferredKinds,
      maxDistancePx,
      projectionMargin: cfg.projectionMargin,
    })

    if (!best) {
      console.warn("OPENING_MATCH_FAILED", {
        phase: "matchOpeningsToWalls",
        openingId: opening.id,
        type: opening.type,
        centerPoint: opening.point,
        widthPx: opening.widthPx,
        preferredWallId: opening.preferredWallId || null,
        preferredKinds: opening.preferredKinds || [],
      })
      unmatched.push(opening)
      return
    }

    const rankedCandidates = (best?.debug?.rankedCandidates || [])
      .slice(0, 2)
      .map((candidate) => ({
        wallId: candidate.wallId,
        wallKind: candidate.wallKind,
        score: roundDebugValue(candidate.score),
        distancePx: roundDebugValue(candidate.distancePx),
      }))
    if (rankedCandidates.length >= 2 && Math.abs(rankedCandidates[0].score - rankedCandidates[1].score) <= 1.5) {
      console.warn("AMBIGUOUS_OPENING_MATCH", {
        openingId: opening.id,
        type: opening.type,
        centerPoint: opening.point,
        widthPx: opening.widthPx,
        candidateScores: rankedCandidates,
      })
    }

    console.log("[matchOpeningsToWalls] opening matched", {
      openingId: opening.id,
      type: opening.type,
      centerPoint: opening.point,
      widthPx: opening.widthPx,
      chosenHostWall: best.wall.id,
      hostWallType: best.wall.kind,
      score: roundDebugValue(best.score),
      rejectedCandidates: best?.debug?.rejectedCandidates || [],
    })

    matched.push({
      ...opening,
      wallId: best.wall.id,
      wallKind: best.wall.kind,
      hostLine: [...best.wall.line],
      point: {
        x: best.projected.x,
        y: best.projected.y,
      },
      t: best.projected.t,
      distancePx: best.distancePx,
      score: best.score,
    })
  })

  return { matched, unmatched }
}

export function splitWallsByOpenings({
  graph,
  matchedOpenings = [],
  options = {},
}) {
  const cfg = { ...graph.options, ...options }
  const grouped = new Map()
  const items = []
  const issues = [...graph.issues]

  matchedOpenings.forEach((opening) => {
    if (!grouped.has(opening.wallId)) grouped.set(opening.wallId, [])
    grouped.get(opening.wallId).push(opening)
  })

  graph.walls.forEach((wall) => {
    const wallOpenings = (grouped.get(wall.id) || [])
      .slice()
      .sort((a, b) => a.t - b.t)

    if (!wallOpenings.length) {
      items.push({
        type: "wall",
        wallKind: wall.kind,
        wallId: wall.id,
        line: [...wall.line],
      })
      return
    }

    const mergedRanges = mergeOpeningRanges(wall, wallOpenings, cfg)
    const wallSegments = carveWallSegments(wall, mergedRanges, cfg)

    console.log("[splitWallsByOpenings] wall split summary", {
      wallId: wall.id,
      wallKind: wall.kind,
      openingCount: wallOpenings.length,
      openingWidthsPx: wallOpenings.map((opening) => roundDebugValue(opening.clearWidthPx || opening.widthPx)),
      mergedRangeCount: mergedRanges.length,
      generatedSegmentCount: wallSegments.length,
    })

    if (!wallSegments.length) {
      issues.push({
        type: "fully-consumed-wall",
        wallId: wall.id,
        line: wall.line,
      })
    }

    wallSegments.forEach((line) => {
      items.push({
        type: "wall",
        wallKind: wall.kind,
        wallId: wall.id,
        line,
      })
    })

    mergedRanges.forEach((range, index) => {
      const centerDist = (range.start + range.end) / 2
      const point = pointAlongWall(wall, centerDist)
      const dominantType = range.typePriority === "door" ? "door" : "window"

      items.push({
        type: dominantType,
        wallKind: wall.kind,
        wallId: wall.id,
        point,
        hostLine: [...wall.line],
        widthPx: range.end - range.start,
        sourceIds: range.sourceIds,
        rangeIndex: index,
      })
    })
  })

  return {
    items,
    issues,
  }
}

export function createWallDebugData({
  graph,
  matchedOpenings = [],
}) {
  return {
    walls: graph.walls.map((wall) => ({
      id: wall.id,
      kind: wall.kind,
      line: wall.line,
      adjacency: wall.adjacentWallIds,
      openingCount: matchedOpenings.filter((item) => item.wallId === wall.id).length,
    })),
    openings: matchedOpenings.map((opening) => ({
      id: opening.id,
      type: opening.type,
      point: opening.point,
      hostLine: opening.hostLine,
      wallId: opening.wallId,
      wallKind: opening.wallKind,
      distancePx: opening.distancePx,
      score: opening.score,
    })),
  }
}

function createWallSegment(id, line, kind, minSegmentLengthPx) {
  const normalized = normalizeLineDirection(line)
  const [x1, y1, x2, y2] = normalized
  const dx = x2 - x1
  const dy = y2 - y1
  const length = Math.hypot(dx, dy)

  if (length < minSegmentLengthPx) return null

  return {
    id,
    kind,
    line: normalized,
    start: { x: x1, y: y1 },
    end: { x: x2, y: y2 },
    length,
    dir: { x: dx / length, y: dy / length },
  }
}

function normalizeLineDirection(line) {
  const [ax, ay, bx, by] = line
  const isForward = ax < bx || (ax === bx && ay <= by)
  return isForward ? [ax, ay, bx, by] : [bx, by, ax, ay]
}

function polygonToSegments(polygon) {
  const segments = []

  for (let i = 0; i < polygon.length; i += 1) {
    const a = polygon[i]
    const b = polygon[(i + 1) % polygon.length]
    if (!a || !b) continue
    segments.push([a[0], a[1], b[0], b[1]])
  }

  return segments
}

function makeOpening(id, type, point, widthPx) {
  const sourceWidth = Math.max(24, point.width || widthPx)
  const clearWidthPx = sourceWidth * (type === "door" ? 0.42 : 0.46)
  const preferredKinds = Array.isArray(point.preferredKinds)
    ? point.preferredKinds.filter((kind) => kind === "outer" || kind === "inner")
    : []
  return {
    id,
    type,
    point: { x: point.x, y: point.y },
    widthPx: sourceWidth,
    clearWidthPx,
    preferredKinds,
    preferredWallId: typeof point.preferredWallId === "string" ? point.preferredWallId : null,
  }
}

function bucketKey(x, y, tol) {
  return `${Math.round(x / tol)}_${Math.round(y / tol)}`
}

function neighboringBucketKeys(x, y, tol) {
  const cx = Math.round(x / tol)
  const cy = Math.round(y / tol)
  const keys = []

  for (let ox = -1; ox <= 1; ox += 1) {
    for (let oy = -1; oy <= 1; oy += 1) {
      keys.push(`${cx + ox}_${cy + oy}`)
    }
  }

  return keys
}

function dedupeWalls(rawWalls, cfg) {
  const kept = []

  rawWalls.forEach((candidate) => {
    const normalized = normalizeLineDirection(candidate.line)
    const [x1, y1, x2, y2] = normalized
    const length = distanceBetweenPoints(x1, y1, x2, y2)
    if (candidate.kind === "inner" && length < cfg.minInnerSegmentLengthPx) return
    const duplicate = kept.some((saved) =>
      areLinesEquivalent(normalized, saved.line, cfg.duplicateWallTolerancePx)
    )

    if (!duplicate) {
      kept.push({
        ...candidate,
        line: normalized,
      })
    }
  })

  return kept
}

function mergeNearParallelInnerWalls(lines, cfg) {
  const normalized = (lines || [])
    .map((line) => normalizeLineDirection(line))
    .filter((line) => distanceBetweenPoints(line[0], line[1], line[2], line[3]) >= cfg.minSegmentLengthPx)

  const horizontal = []
  const vertical = []
  const angled = []

  normalized.forEach((line) => {
    const [x1, y1, x2, y2] = line
    const dx = Math.abs(x2 - x1)
    const dy = Math.abs(y2 - y1)
    if (dx >= dy * cfg.axisSkewRatio) {
      horizontal.push(line)
      return
    }
    if (dy >= dx * cfg.axisSkewRatio) {
      vertical.push(line)
      return
    }
    angled.push(line)
  })

  const mergedHorizontal = mergeAxisAligned(horizontal, "horizontal", cfg)
  const mergedVertical = mergeAxisAligned(vertical, "vertical", cfg)

  return [...mergedHorizontal, ...mergedVertical, ...angled]
}

function mergeAxisAligned(lines, axis, cfg) {
  const groups = []
  const isHorizontal = axis === "horizontal"

  lines.forEach((line) => {
    const fixed = isHorizontal ? (line[1] + line[3]) * 0.5 : (line[0] + line[2]) * 0.5
    let group = groups.find((item) => Math.abs(item.fixed - fixed) <= cfg.parallelMergeDistancePx)
    if (!group) {
      group = { fixed, intervals: [], fixedSamples: [] }
      groups.push(group)
    }
    group.fixedSamples.push(fixed)
    if (isHorizontal) {
      group.intervals.push([Math.min(line[0], line[2]), Math.max(line[0], line[2])])
    } else {
      group.intervals.push([Math.min(line[1], line[3]), Math.max(line[1], line[3])])
    }
  })

  const mergedLines = []

  groups.forEach((group) => {
    const fixed = average(group.fixedSamples)
    const mergedIntervals = mergeIntervals(group.intervals, cfg.collinearGapPx)
    mergedIntervals.forEach(([start, end]) => {
      if (end - start < cfg.minInnerSegmentLengthPx) return
      if (isHorizontal) {
        mergedLines.push([start, fixed, end, fixed])
      } else {
        mergedLines.push([fixed, start, fixed, end])
      }
    })
  })

  return mergedLines
}

function mergeIntervals(intervals, gap) {
  if (!intervals.length) return []
  const sorted = intervals.slice().sort((a, b) => a[0] - b[0])
  const merged = [[sorted[0][0], sorted[0][1]]]

  for (let i = 1; i < sorted.length; i += 1) {
    const [start, end] = sorted[i]
    const last = merged[merged.length - 1]
    if (start <= last[1] + gap) {
      last[1] = Math.max(last[1], end)
    } else {
      merged.push([start, end])
    }
  }

  return merged
}

function average(values) {
  if (!values.length) return 0
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function splitWallsAtIntersections(walls, cfg) {
  const cutsById = new Map()
  walls.forEach((wall) => {
    const length = distanceBetweenPoints(wall.line[0], wall.line[1], wall.line[2], wall.line[3])
    cutsById.set(wall.id, [0, length])
  })

  for (let i = 0; i < walls.length; i += 1) {
    for (let j = i + 1; j < walls.length; j += 1) {
      const a = walls[i]
      const b = walls[j]
      const hit = segmentIntersection(a.line, b.line, cfg.splitIntersectionTolerancePx)
      if (!hit) continue

      const aLen = distanceBetweenPoints(a.line[0], a.line[1], a.line[2], a.line[3])
      const bLen = distanceBetweenPoints(b.line[0], b.line[1], b.line[2], b.line[3])

      if (hit.ta > 0.01 && hit.ta < 0.99) cutsById.get(a.id).push(hit.ta * aLen)
      if (hit.tb > 0.01 && hit.tb < 0.99) cutsById.get(b.id).push(hit.tb * bLen)
    }
  }

  const splitWalls = []

  walls.forEach((wall) => {
    const [x1, y1, x2, y2] = wall.line
    const dx = x2 - x1
    const dy = y2 - y1
    const length = Math.hypot(dx, dy)
    const dirX = length > 0 ? dx / length : 0
    const dirY = length > 0 ? dy / length : 0
    const cuts = uniqueSorted(cutsById.get(wall.id) || [])

    if (cuts.length < 2) {
      splitWalls.push(wall)
      return
    }

    let segmentIndex = 0
    for (let i = 0; i < cuts.length - 1; i += 1) {
      const start = cuts[i]
      const end = cuts[i + 1]
      if (end - start < cfg.minSegmentLengthPx) continue
      const sx = x1 + dirX * start
      const sy = y1 + dirY * start
      const ex = x1 + dirX * end
      const ey = y1 + dirY * end
      splitWalls.push({
        id: `${wall.id}-s${segmentIndex}`,
        kind: wall.kind,
        line: normalizeLineDirection([sx, sy, ex, ey]),
      })
      segmentIndex += 1
    }
  })

  return splitWalls
}

function segmentIntersection(lineA, lineB, tolerancePx) {
  const [x1, y1, x2, y2] = lineA
  const [x3, y3, x4, y4] = lineB
  const denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
  if (Math.abs(denom) < 1e-6) return null

  const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
  const u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
  if (t < 0 || t > 1 || u < 0 || u > 1) return null

  const px = x1 + t * (x2 - x1)
  const py = y1 + t * (y2 - y1)

  const onA = projectPointToSegment(px, py, lineA)
  const onB = projectPointToSegment(px, py, lineB)
  if (onA.distancePx > tolerancePx || onB.distancePx > tolerancePx) return null

  return { x: px, y: py, ta: t, tb: u }
}

function uniqueSorted(values) {
  const sorted = values.slice().sort((a, b) => a - b)
  const unique = []
  sorted.forEach((value) => {
    const last = unique[unique.length - 1]
    if (last == null || Math.abs(value - last) > 0.001) {
      unique.push(value)
    }
  })
  return unique
}

function areLinesEquivalent(a, b, tol) {
  return (
    distanceBetweenPoints(a[0], a[1], b[0], b[1]) <= tol &&
    distanceBetweenPoints(a[2], a[3], b[2], b[3]) <= tol
  )
}

function distanceBetweenPoints(ax, ay, bx, by) {
  return Math.hypot(bx - ax, by - ay)
}

function findBestCandidate(
  opening,
  walls,
  {
    allowedKinds,
    maxDistancePx,
    projectionMargin,
  }
) {
  let best = null
  const rejectedCandidates = []
  const rankedCandidates = []

  walls.forEach((wall) => {
    if (allowedKinds?.length && !allowedKinds.includes(wall.kind)) {
      rejectedCandidates.push({
        wallId: wall.id,
        wallKind: wall.kind,
        reason: "kind-mismatch",
      })
      return
    }

    const projection = projectPointToSegment(opening.point.x, opening.point.y, wall.line)

    if (!projection.insideSegment) {
      rejectedCandidates.push({
        wallId: wall.id,
        wallKind: wall.kind,
        reason: "outside-segment",
        t: roundDebugValue(projection.t),
        distancePx: roundDebugValue(projection.distancePx),
      })
      return
    }
    if (projection.t <= projectionMargin || projection.t >= 1 - projectionMargin) {
      rejectedCandidates.push({
        wallId: wall.id,
        wallKind: wall.kind,
        reason: "projection-margin",
        t: roundDebugValue(projection.t),
        projectionMargin: roundDebugValue(projectionMargin),
      })
      return
    }
    if (projection.distancePx > maxDistancePx) {
      rejectedCandidates.push({
        wallId: wall.id,
        wallKind: wall.kind,
        reason: "distance-too-large",
        distancePx: roundDebugValue(projection.distancePx),
        maxDistancePx: roundDebugValue(maxDistancePx),
      })
      return
    }
    const openingWidth = opening.clearWidthPx || opening.widthPx
    if (openingWidth >= wall.length - 4) {
      rejectedCandidates.push({
        wallId: wall.id,
        wallKind: wall.kind,
        reason: "opening-too-wide-for-wall",
        openingWidth: roundDebugValue(openingWidth),
        wallLength: roundDebugValue(wall.length),
      })
      return
    }

    const endClearancePx = Math.min(
      projection.distanceAlong,
      wall.length - projection.distanceAlong
    )

    const requiredClearancePx =
      opening.type === "door"
        ? openingWidth * 0.45 + 6
        : openingWidth * 0.35 + 4
    if (endClearancePx < requiredClearancePx) {
      rejectedCandidates.push({
        wallId: wall.id,
        wallKind: wall.kind,
        reason: "insufficient-end-clearance",
        endClearancePx: roundDebugValue(endClearancePx),
        requiredClearancePx: roundDebugValue(requiredClearancePx),
      })
      return
    }

    const centerBias = Math.abs(0.5 - projection.t) * 4
    const edgePenalty = Math.max(0, openingWidth * 0.5 - endClearancePx) * 0.35
    const kindPenalty =
      opening.type === "door" && wall.kind === "outer" && allowedKinds?.[0] === "inner"
        ? 2
        : 0

    const score = projection.distancePx + centerBias + edgePenalty + kindPenalty
    rankedCandidates.push({
      wallId: wall.id,
      wallKind: wall.kind,
      score,
      distancePx: projection.distancePx,
      t: projection.t,
      endClearancePx,
    })

    if (!best || score < best.score) {
      best = {
        wall,
        projected: {
          x: projection.x,
          y: projection.y,
          t: projection.t,
          distanceAlong: projection.distanceAlong,
        },
        distancePx: projection.distancePx,
        score,
      }
    }
  })

  rankedCandidates.sort((a, b) => a.score - b.score)
  if (best) {
    best.debug = {
      rankedCandidates: rankedCandidates.map((candidate) => ({
        ...candidate,
        score: roundDebugValue(candidate.score),
        distancePx: roundDebugValue(candidate.distancePx),
        t: roundDebugValue(candidate.t),
        endClearancePx: roundDebugValue(candidate.endClearancePx),
      })),
      rejectedCandidates,
    }
  }

  return best
}

function roundDebugValue(value) {
  return Number.isFinite(value) ? Number(value.toFixed(3)) : value
}

function projectPointToSegment(px, py, line) {
  const [x1, y1, x2, y2] = line
  const dx = x2 - x1
  const dy = y2 - y1
  const lenSq = dx * dx + dy * dy

  if (lenSq < 1e-9) {
    return {
      x: x1,
      y: y1,
      t: 0,
      insideSegment: false,
      distancePx: Math.hypot(px - x1, py - y1),
      distanceAlong: 0,
    }
  }

  const tRaw = ((px - x1) * dx + (py - y1) * dy) / lenSq
  const x = x1 + tRaw * dx
  const y = y1 + tRaw * dy
  const distancePx = Math.hypot(px - x, py - y)
  const length = Math.sqrt(lenSq)

  return {
    x,
    y,
    t: tRaw,
    insideSegment: tRaw >= 0 && tRaw <= 1,
    distancePx,
    distanceAlong: tRaw * length,
  }
}

function mergeOpeningRanges(wall, openings, cfg) {
  const ranges = openings
    .map((opening) => {
      const center = opening.t * wall.length
      const half = (opening.clearWidthPx || opening.widthPx) / 2

      return {
        start: Math.max(0, center - half),
        end: Math.min(wall.length, center + half),
        sourceIds: [opening.id],
        typePriority: opening.type === "door" ? "door" : "window",
      }
    })
    .sort((a, b) => a.start - b.start)

  const merged = []

  ranges.forEach((range) => {
    const last = merged[merged.length - 1]

    if (!last) {
      merged.push({ ...range })
      return
    }

    const overlap = range.start <= last.end
    const closeSameType =
      range.typePriority === last.typePriority &&
      range.start <= last.end + cfg.overlapJoinGapPx

    if (overlap || closeSameType) {
      last.end = Math.max(last.end, range.end)
      last.sourceIds.push(...range.sourceIds)
      if (range.typePriority === "door") {
        last.typePriority = "door"
      }
      return
    }

    merged.push({ ...range })
  })

  if (merged[0] && merged[0].start < cfg.minEdgePiecePx) {
    merged[0].start = 0
  }

  const lastRange = merged[merged.length - 1]
  if (lastRange && wall.length - lastRange.end < cfg.minEdgePiecePx) {
    lastRange.end = wall.length
  }

  for (let i = 1; i < merged.length; i += 1) {
    const prev = merged[i - 1]
    const curr = merged[i]
    const gap = curr.start - prev.end

    if (gap > 0 && gap < cfg.minWallPiecePx) {
      const mid = (prev.end + curr.start) / 2
      prev.end = mid
      curr.start = mid
    }
  }

  return merged.filter((range) => range.end - range.start > 2)
}

function carveWallSegments(wall, ranges, cfg) {
  const segments = []
  let cursor = 0

  ranges.forEach((range) => {
    if (range.start - cursor >= cfg.minWallPiecePx) {
      segments.push(lineSlice(wall, cursor, range.start))
    }
    cursor = Math.max(cursor, range.end)
  })

  if (wall.length - cursor >= cfg.minWallPiecePx) {
    segments.push(lineSlice(wall, cursor, wall.length))
  }

  return segments
}

function lineSlice(wall, startDist, endDist) {
  const sx = wall.start.x + wall.dir.x * startDist
  const sy = wall.start.y + wall.dir.y * startDist
  const ex = wall.start.x + wall.dir.x * endDist
  const ey = wall.start.y + wall.dir.y * endDist

  return [sx, sy, ex, ey]
}

function pointAlongWall(wall, dist) {
  return {
    x: wall.start.x + wall.dir.x * dist,
    y: wall.start.y + wall.dir.y * dist,
  }
}

param(
  [string]$GroundTruth = "backend/ground_truth/floor_1_clean.json",
  [string]$ApiBase = "http://127.0.0.1:8010",
  [double]$FloorHeight = 3.2
)

$ErrorActionPreference = "Stop"

function Get-BBox {
  param([array]$Polygon)
  if (-not $Polygon -or $Polygon.Count -eq 0) {
    return $null
  }
  $xs = @()
  $ys = @()
  foreach ($pt in $Polygon) {
    $xs += [double]$pt[0]
    $ys += [double]$pt[1]
  }
  return [ordered]@{
    min_x = ($xs | Measure-Object -Minimum).Minimum
    max_x = ($xs | Measure-Object -Maximum).Maximum
    min_y = ($ys | Measure-Object -Minimum).Minimum
    max_y = ($ys | Measure-Object -Maximum).Maximum
    width = (($xs | Measure-Object -Maximum).Maximum - ($xs | Measure-Object -Minimum).Minimum)
    height = (($ys | Measure-Object -Maximum).Maximum - ($ys | Measure-Object -Minimum).Minimum)
  }
}

function Normalize-PointToTargetBBox {
  param(
    [double]$X,
    [double]$Y,
    [hashtable]$SourceBBox,
    [hashtable]$TargetBBox
  )
  if (-not $SourceBBox -or -not $TargetBBox -or [double]$SourceBBox.width -le 0 -or [double]$SourceBBox.height -le 0) {
    return @($X, $Y)
  }
  $scaleX = [double]$TargetBBox.width / [double]$SourceBBox.width
  $scaleY = [double]$TargetBBox.height / [double]$SourceBBox.height
  $normalizedX = [double]$TargetBBox.min_x + (($X - [double]$SourceBBox.min_x) * $scaleX)
  $normalizedY = [double]$TargetBBox.min_y + (($Y - [double]$SourceBBox.min_y) * $scaleY)
  return @($normalizedX, $normalizedY)
}

function Normalize-LineListToTargetBBox {
  param(
    [array]$Lines,
    [hashtable]$SourceBBox,
    [hashtable]$TargetBBox
  )
  $normalized = @()
  foreach ($line in $Lines) {
    $p1 = Normalize-PointToTargetBBox -X ([double]$line[0]) -Y ([double]$line[1]) -SourceBBox $SourceBBox -TargetBBox $TargetBBox
    $p2 = Normalize-PointToTargetBBox -X ([double]$line[2]) -Y ([double]$line[3]) -SourceBBox $SourceBBox -TargetBBox $TargetBBox
    $normalized += ,@($p1[0], $p1[1], $p2[0], $p2[1])
  }
  return $normalized
}

function Normalize-OpeningListToTargetBBox {
  param(
    [array]$Openings,
    [hashtable]$SourceBBox,
    [hashtable]$TargetBBox
  )
  $normalized = @()
  if (-not $SourceBBox -or -not $TargetBBox -or [double]$SourceBBox.width -le 0 -or [double]$SourceBBox.height -le 0) {
    return $Openings
  }
  $scaleX = [double]$TargetBBox.width / [double]$SourceBBox.width
  $scaleY = [double]$TargetBBox.height / [double]$SourceBBox.height
  $widthScale = ($scaleX + $scaleY) / 2.0
  foreach ($opening in $Openings) {
    $pt = Normalize-PointToTargetBBox -X ([double]$opening.x) -Y ([double]$opening.y) -SourceBBox $SourceBBox -TargetBBox $TargetBBox
    $normalized += ,([ordered]@{
      x = [math]::Round($pt[0], 3)
      y = [math]::Round($pt[1], 3)
      width = [math]::Round(([double]$opening.width * $widthScale), 3)
    })
  }
  return $normalized
}

function Get-LineAngle {
  param([array]$Line)
  $dx = [double]$Line[2] - [double]$Line[0]
  $dy = [double]$Line[3] - [double]$Line[1]
  $angle = [math]::Abs(([math]::Atan2($dy, $dx) * 180.0 / [math]::PI)) % 180.0
  return $angle
}

function Get-EndpointPairDistance {
  param([array]$A, [array]$B)
  $forward =
    [math]::Sqrt(([double]$A[0] - [double]$B[0]) * ([double]$A[0] - [double]$B[0]) + ([double]$A[1] - [double]$B[1]) * ([double]$A[1] - [double]$B[1])) +
    [math]::Sqrt(([double]$A[2] - [double]$B[2]) * ([double]$A[2] - [double]$B[2]) + ([double]$A[3] - [double]$B[3]) * ([double]$A[3] - [double]$B[3]))
  $reverse =
    [math]::Sqrt(([double]$A[0] - [double]$B[2]) * ([double]$A[0] - [double]$B[2]) + ([double]$A[1] - [double]$B[3]) * ([double]$A[1] - [double]$B[3])) +
    [math]::Sqrt(([double]$A[2] - [double]$B[0]) * ([double]$A[2] - [double]$B[0]) + ([double]$A[3] - [double]$B[1]) * ([double]$A[3] - [double]$B[1]))
  return [math]::Min($forward, $reverse) / 2.0
}

function Get-AxisSignature {
  param([array]$Line)
  $dx = [math]::Abs([double]$Line[2] - [double]$Line[0])
  $dy = [math]::Abs([double]$Line[3] - [double]$Line[1])
  if ($dx -ge $dy) {
    return @{
      axis = "x"
      fixed = (([double]$Line[1] + [double]$Line[3]) / 2.0)
      start = [math]::Min([double]$Line[0], [double]$Line[2])
      end = [math]::Max([double]$Line[0], [double]$Line[2])
    }
  }
  return @{
    axis = "y"
    fixed = (([double]$Line[0] + [double]$Line[2]) / 2.0)
    start = [math]::Min([double]$Line[1], [double]$Line[3])
    end = [math]::Max([double]$Line[1], [double]$Line[3])
  }
}

function Get-LineOverlapRatio {
  param([array]$A, [array]$B)
  $sigA = Get-AxisSignature $A
  $sigB = Get-AxisSignature $B
  if ($sigA.axis -ne $sigB.axis) {
    return 0.0
  }
  $overlap = [math]::Max(0.0, [math]::Min($sigA.end, $sigB.end) - [math]::Max($sigA.start, $sigB.start))
  $shorter = [math]::Min($sigA.end - $sigA.start, $sigB.end - $sigB.start)
  if ($shorter -le 0) {
    return 0.0
  }
  return $overlap / $shorter
}

function Get-LinePerpendicularDistance {
  param([array]$A, [array]$B)
  $sigA = Get-AxisSignature $A
  $sigB = Get-AxisSignature $B
  if ($sigA.axis -ne $sigB.axis) {
    return [double]::PositiveInfinity
  }
  return [math]::Abs($sigA.fixed - $sigB.fixed)
}

function Convert-LineToObject {
  param([array]$Line)
  return [ordered]@{
    x1 = [double]$Line[0]
    y1 = [double]$Line[1]
    x2 = [double]$Line[2]
    y2 = [double]$Line[3]
  }
}

function Compare-Lines {
  param(
    [array]$Expected,
    [array]$Actual,
    [double]$EndpointTol = 24.0,
    [double]$AngleTol = 8.0,
    [double]$PerpTol = 16.0,
    [double]$OverlapMin = 0.45
  )

  $unmatchedActual = New-Object System.Collections.Generic.HashSet[int]
  for ($i = 0; $i -lt $Actual.Count; $i++) {
    [void]$unmatchedActual.Add($i)
  }

  $missing = @()
  $matchedCount = 0

  foreach ($expectedLine in $Expected) {
    $bestIdx = $null
    $bestScore = $null
    $expectedAngle = Get-LineAngle $expectedLine
    foreach ($idx in @($unmatchedActual)) {
      $actualLine = $Actual[$idx]
      $actualAngle = Get-LineAngle $actualLine
      $angleDiff = [math]::Abs($expectedAngle - $actualAngle)
      $angleDiff = [math]::Min($angleDiff, 180.0 - $angleDiff)
      $overlap = Get-LineOverlapRatio $expectedLine $actualLine
      $perp = Get-LinePerpendicularDistance $expectedLine $actualLine
      $endpoint = Get-EndpointPairDistance $expectedLine $actualLine
      if ($angleDiff -gt $AngleTol -or $perp -gt $PerpTol -or $overlap -lt $OverlapMin -or $endpoint -gt $EndpointTol) {
        continue
      }
      $score = $endpoint + $perp + ((1.0 - $overlap) * 10.0)
      if ($null -eq $bestScore -or $score -lt $bestScore) {
        $bestScore = $score
        $bestIdx = $idx
      }
    }

    if ($null -eq $bestIdx) {
      $missing += ,(Convert-LineToObject $expectedLine)
      continue
    }

    [void]$unmatchedActual.Remove($bestIdx)
    $matchedCount += 1
  }

  $extra = @()
  foreach ($idx in @($unmatchedActual)) {
    $extra += ,(Convert-LineToObject $Actual[$idx])
  }

  return [ordered]@{
    matched_count = $matchedCount
    missing = $missing
    extra = $extra
  }
}

function Compare-Openings {
  param(
    [array]$Expected,
    [array]$Actual,
    [double]$CenterTol = 26.0,
    [double]$WidthTol = 26.0
  )

  $unmatchedActual = New-Object System.Collections.Generic.HashSet[int]
  for ($i = 0; $i -lt $Actual.Count; $i++) {
    [void]$unmatchedActual.Add($i)
  }

  $missing = @()
  $matchedCount = 0

  foreach ($expectedItem in $Expected) {
    $bestIdx = $null
    $bestScore = $null
    foreach ($idx in @($unmatchedActual)) {
      $actualItem = $Actual[$idx]
      $dx = [double]$expectedItem.x - [double]$actualItem.x
      $dy = [double]$expectedItem.y - [double]$actualItem.y
      $centerDist = [math]::Sqrt(($dx * $dx) + ($dy * $dy))
      $widthDist = [math]::Abs([double]$expectedItem.width - [double]$actualItem.width)
      if ($centerDist -gt $CenterTol -or $widthDist -gt $WidthTol) {
        continue
      }
      $score = $centerDist + ($widthDist * 0.3)
      if ($null -eq $bestScore -or $score -lt $bestScore) {
        $bestScore = $score
        $bestIdx = $idx
      }
    }

    if ($null -eq $bestIdx) {
      $missing += ,$expectedItem
      continue
    }

    [void]$unmatchedActual.Remove($bestIdx)
    $matchedCount += 1
  }

  $extra = @()
  foreach ($idx in @($unmatchedActual)) {
    $extra += ,$Actual[$idx]
  }

  return [ordered]@{
    matched_count = $matchedCount
    missing = $missing
    extra = $extra
  }
}

$groundTruthPath = Join-Path (Get-Location) $GroundTruth
$gt = Get-Content $groundTruthPath -Raw | ConvertFrom-Json
$sourceImage = Join-Path (Get-Location) $gt.source_image

if (-not (Test-Path $sourceImage)) {
  throw "Source image not found: $sourceImage"
}

$health = Invoke-RestMethod "$ApiBase/health"
if ($health.status -ne "ok") {
  throw "Backend health not ok"
}

$uploadJson = curl.exe -s -X POST "$ApiBase/projects/upload" -F "files=@$sourceImage" -F "floor_count=1"
if (-not $uploadJson) {
  throw "Upload returned empty response"
}
$upload = $uploadJson | ConvertFrom-Json
$projectId = $upload.project_id
if (-not $projectId) {
  throw "Upload response missing project_id"
}

$analysis = Invoke-RestMethod "$ApiBase/projects/${projectId}?floor_height=$FloorHeight"
$floor = $analysis.floors[0]

$expectedBBox = Get-BBox $gt.polygon
$actualBBox = Get-BBox $floor.polygon
$normalizedActualLines = $floor.inner_walls
$normalizedActualDoors = $floor.doors
$normalizedActualWindows = $floor.windows
$bboxDiff = $null
if ($expectedBBox -and $actualBBox) {
  $bboxDiff = [ordered]@{}
  foreach ($key in @("min_x", "max_x", "min_y", "max_y", "width", "height")) {
    $bboxDiff[$key] = [math]::Round(([double]$actualBBox[$key] - [double]$expectedBBox[$key]), 3)
  }
  $normalizedActualLines = Normalize-LineListToTargetBBox -Lines $floor.inner_walls -SourceBBox $actualBBox -TargetBBox $expectedBBox
  $normalizedActualDoors = Normalize-OpeningListToTargetBBox -Openings $floor.doors -SourceBBox $actualBBox -TargetBBox $expectedBBox
  $normalizedActualWindows = Normalize-OpeningListToTargetBBox -Openings $floor.windows -SourceBBox $actualBBox -TargetBBox $expectedBBox
}

$wallReport = Compare-Lines -Expected $gt.inner_walls -Actual $normalizedActualLines
$doorReport = Compare-Openings -Expected $gt.doors -Actual $normalizedActualDoors
$windowReport = Compare-Openings -Expected $gt.windows -Actual $normalizedActualWindows

$report = [ordered]@{
  floor_name = $gt.floor_name
  project_id = $projectId
  polygon = [ordered]@{
    expected_exists = [bool]($gt.polygon.Count -gt 0)
    actual_exists = [bool]($floor.polygon.Count -gt 0)
    expected_point_count = $gt.polygon.Count
    actual_point_count = $floor.polygon.Count
    expected_bbox = $expectedBBox
    actual_bbox = $actualBBox
    comparison_space = "actual geometry normalized into ground-truth bbox for walls/openings matching"
    bbox_diff = $bboxDiff
  }
  inner_walls = [ordered]@{
    expected_count = $gt.inner_walls.Count
    actual_count = $floor.inner_walls.Count
    missing_inner_walls = $wallReport.missing
    extra_inner_walls = $wallReport.extra
    matched_count = $wallReport.matched_count
  }
  doors = [ordered]@{
    expected_count = $gt.doors.Count
    actual_count = $floor.doors.Count
    missing_doors = $doorReport.missing
    extra_doors = $doorReport.extra
    matched_count = $doorReport.matched_count
  }
  windows = [ordered]@{
    expected_count = $gt.windows.Count
    actual_count = $floor.windows.Count
    missing_windows = $windowReport.missing
    extra_windows = $windowReport.extra
    matched_count = $windowReport.matched_count
  }
  rooms = [ordered]@{
    expected_room_count = [int]$gt.room_count
    actual_room_count = $floor.rooms.Count
    room_count_diff = ($floor.rooms.Count - [int]$gt.room_count)
  }
}

Write-Host "Comparison floor: $($report.floor_name)"
Write-Host "Project ID: $projectId"
Write-Host "Polygon bbox diff: $(($report.polygon.bbox_diff | ConvertTo-Json -Compress))"
Write-Host "Inner walls: expected=$($report.inner_walls.expected_count) actual=$($report.inner_walls.actual_count) missing=$($report.inner_walls.missing_inner_walls.Count) extra=$($report.inner_walls.extra_inner_walls.Count)"
Write-Host "Doors: expected=$($report.doors.expected_count) actual=$($report.doors.actual_count) missing=$($report.doors.missing_doors.Count) extra=$($report.doors.extra_doors.Count)"
Write-Host "Windows: expected=$($report.windows.expected_count) actual=$($report.windows.actual_count) missing=$($report.windows.missing_windows.Count) extra=$($report.windows.extra_windows.Count)"
Write-Host "Rooms: expected=$($report.rooms.expected_room_count) actual=$($report.rooms.actual_room_count) diff=$($report.rooms.room_count_diff)"
Write-Output ($report | ConvertTo-Json -Depth 8)

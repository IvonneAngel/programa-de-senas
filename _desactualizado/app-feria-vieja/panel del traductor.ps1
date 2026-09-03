param(
  [switch]$Preview,
  [switch]$Web
)

$ErrorActionPreference = "Stop"

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$script:Esc = [char]27
$script:N = [char]0x00F1
$script:Aa = [char]0x00E1
$script:Ee = [char]0x00E9
$script:Ii = [char]0x00ED
$script:Oo = [char]0x00F3
$script:Uu = [char]0x00FA
$script:Senias = "se" + $script:N + "as"
$script:Camara = "c" + $script:Aa + "mara"
$script:Practica = "pr" + $script:Aa + "ctica"
$script:Informacion = "Informaci" + $script:Oo + "n"
$script:Creditos = "Cr" + $script:Ee + "ditos"
$script:Tecnicas = "T" + $script:Ee + "cnicas"
$script:Ingenierias = "Ingenier" + $script:Ii + "as"
$script:Educacion = "Educaci" + $script:Oo + "n"
$script:Secretaria = "Secretar" + $script:Ii + "a"
$script:Tecnologia = "Tecnolog" + $script:Ii + "a"

$script:Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$script:Codigo = Join-Path $script:Root "codigo"
$script:PowerShellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$script:Images = Join-Path $script:Root "imagenes de entrenamiento"
$script:ProcessedData = Join-Path $script:Root "datos procesados del modelo"
$script:ExternalBase = Join-Path (Join-Path $env:USERPROFILE "Downloads") ("proyecto de " + $script:Senias + " archivos externos")
$script:TestsBase = Join-Path (Join-Path $env:USERPROFILE "Downloads") ("proyecto de " + $script:Senias + " pruebas")
$script:Reports = Join-Path $script:ExternalBase "reportes"
$script:VenvPython = Join-Path $script:ExternalBase "entorno python\.venv\Scripts\python.exe"
$script:TranslatorScript = Join-Path $script:Codigo ("aplicacion\traductor_de_" + $script:Senias + ".py")
$script:InterfaceServer = Join-Path $script:Codigo "aplicacion\servidor_interfaz.py"
$script:InterfaceDist = Join-Path $script:Codigo "interfaz ui\dist\index.html"
$script:PrepareScript = Join-Path $script:Codigo "herramientas\preparar_entorno.py"
$script:ModelKeras = Join-Path $script:Codigo "modelos\sign_model.keras"
$script:ModelTflite = Join-Path $script:Codigo "modelos\sign_model.tflite"
$script:ModelQuality = Join-Path $script:Codigo "modelos\model_quality.json"

[Console]::Title = "Traductor de " + $script:Senias + " IA"
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:GLOG_minloglevel = "3"
$env:ABSL_LOGGING_MIN_LOG_LEVEL = "3"

function Use-Color($code, $text) {
  return "$script:Esc[$code`m$text$script:Esc[0m"
}

function Clear-Panel {
  Write-Host -NoNewline "$script:Esc[H$script:Esc[J"
}

function Write-Line($text = "") {
  Write-Host $text
}

$script:PanelWidth = 72
$script:PanelLeft = 4

function Get-TerminalWidth {
  try {
    if ([Console]::WindowWidth -gt 0) {
      return [Console]::WindowWidth
    }
  } catch {
  }
  return 80
}

function Get-TerminalHeight {
  try {
    if ([Console]::WindowHeight -gt 0) {
      return [Console]::WindowHeight
    }
  } catch {
  }
  return 30
}

function Use-CompactVerticalLayout {
  return (Get-TerminalHeight) -lt 38
}

function Get-TopPadding {
  $height = Get-TerminalHeight
  if ($height -ge 48) {
    return 3
  }
  if ($height -ge 40) {
    return 2
  }
  return 1
}

function Get-Indent([int]$contentWidth = $script:PanelWidth) {
  $width = Get-TerminalWidth
  $targetWidth = [Math]::Min($contentWidth, [Math]::Max(1, $width - 2))
  $spaces = [Math]::Max(1, [int](($width - $targetWidth) / 2))
  return " " * $spaces
}

function Write-PanelLine($text = "", [int]$contentWidth = $script:PanelWidth) {
  Write-Line ((Get-Indent $contentWidth) + $text)
}

function Get-CursorTopSafe {
  try {
    return [Console]::CursorTop
  } catch {
  }
  return -1
}

function Set-CursorSafe([int]$left, [int]$top) {
  try {
    [Console]::SetCursorPosition($left, $top)
    return $true
  } catch {
  }
  return $false
}

function Write-LogoLine($colored, $visible) {
  Write-Line ((Get-Indent $visible.Length) + $colored)
}

function Wait-Key($label = "Presiona cualquier tecla para volver") {
  Write-Line ""
  Write-Host -NoNewline ((Get-Indent) + (Use-Color "90" "$label..."))
  [void]$Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

function Status-Text($path) {
  if (Test-Path $path) {
    return Use-Color "92" "listo"
  }
  return Use-Color "91" "falta"
}

function U($hex) {
  return [char][Convert]::ToInt32($hex, 16)
}

function Use-Rgb([int]$r, [int]$g, [int]$b, $text) {
  return "$script:Esc[38;2;$r;$g;$b`m$text$script:Esc[0m"
}

function Get-FemeciBlockLetters {
  return @(
    @{ Width = 6; Rows = @("111111", "110000", "110000", "111110", "110000", "110000", "110000") },
    @{ Width = 6; Rows = @("111111", "110000", "110000", "111110", "110000", "110000", "111111") },
    @{ Width = 8; Rows = @("11000011", "11100111", "11111111", "11011011", "11000011", "11000011", "11000011") },
    @{ Width = 6; Rows = @("111111", "110000", "110000", "111110", "110000", "110000", "111111") },
    @{ Width = 6; Rows = @("011111", "110000", "110000", "110000", "110000", "110000", "011111") },
    @{ Width = 4; Rows = @("1111", "0110", "0110", "0110", "0110", "0110", "1111") }
  )
}

function New-FemeciBlockGrid {
  $filled = @{}
  $letters = @{}
  $shadow = @{}
  $shadowLetters = @{}
  $coords = @()
  $bounds = @()
  $cursor = 0
  $height = 7
  $letterIndex = 0
  foreach ($letter in (Get-FemeciBlockLetters)) {
    $bounds += @{
      Start = $cursor
      End = $cursor + $letter.Width - 1
      Index = $letterIndex
    }
    for ($row = 0; $row -lt $height; $row++) {
      for ($col = 0; $col -lt $letter.Width; $col++) {
        if ($letter.Rows[$row][$col] -eq "1") {
          $filled["$row,$($cursor + $col)"] = $true
          $letters["$row,$($cursor + $col)"] = $letterIndex
          $coords += @{
            Row = $row
            Col = $cursor + $col
            Letter = $letterIndex
          }
        }
      }
    }
    $cursor += $letter.Width + 2
    $letterIndex += 1
  }

  foreach ($coord in $coords) {
    $offsets = @(@(0, 1))
    if ($coord.Row -eq ($height - 1)) {
      $offsets += ,@(1, 1)
    }
    foreach ($offset in $offsets) {
      $shadowRow = $coord.Row + $offset[0]
      $shadowCol = $coord.Col + $offset[1]
      $key = "$shadowRow,$shadowCol"
      if (-not $filled.ContainsKey($key)) {
        $shadow[$key] = $true
        $shadowLetters[$key] = $coord.Letter
      }
    }
  }

  return @{
    Filled = $filled
    Letters = $letters
    Shadow = $shadow
    ShadowLetters = $shadowLetters
    Bounds = $bounds
    Height = $height + 1
    FrontHeight = $height
    Width = $cursor
  }
}

function Test-FemeciCell($filled, [int]$row, [int]$col) {
  return $filled.ContainsKey("$row,$col")
}

function Test-FemeciColumnInsideLetter($bounds, [int]$col) {
  foreach ($bound in $bounds) {
    if ($col -ge $bound.Start -and $col -le $bound.End) {
      return $true
    }
  }
  return $false
}

function Limit-Rgb([int]$value) {
  return [Math]::Min(255, [Math]::Max(0, $value))
}

function Shift-Rgb($rgb, [int]$delta) {
  return @(
    (Limit-Rgb ($rgb[0] + $delta)),
    (Limit-Rgb ($rgb[1] + $delta)),
    (Limit-Rgb ($rgb[2] + $delta))
  )
}

function Render-TexturedFrontCell($rgb, [int]$row, [int]$col, [bool]$wideMode, [bool]$topEdge, [bool]$leftEdge, [bool]$rightEdge, [bool]$bottomEdge) {
  $solid = [string](U "2588")
  $dense = [string](U "2593")
  $medium = [string](U "2592")
  $light = [string](U "2591")

  if ($topEdge -or $leftEdge) {
    $hi = Shift-Rgb $rgb 24
    $mid = Shift-Rgb $rgb 4
    if ($wideMode) {
      return (Use-Rgb $hi[0] $hi[1] $hi[2] $solid) + (Use-Rgb $mid[0] $mid[1] $mid[2] $dense)
    }
    return Use-Rgb $hi[0] $hi[1] $hi[2] $solid
  }

  if ($rightEdge -or $bottomEdge) {
    $dark = Shift-Rgb $rgb -34
    $deep = Shift-Rgb $rgb -52
    if ($wideMode) {
      return (Use-Rgb $dark[0] $dark[1] $dark[2] $dense) + (Use-Rgb $deep[0] $deep[1] $deep[2] $medium)
    }
    return Use-Rgb $dark[0] $dark[1] $dark[2] $dense
  }

  $pattern = (($row * 7) + ($col * 5)) % 6
  switch ($pattern) {
    0 {
      $hi = Shift-Rgb $rgb 14
      if ($wideMode) {
        return (Use-Rgb $hi[0] $hi[1] $hi[2] $solid) + (Use-Rgb $rgb[0] $rgb[1] $rgb[2] $solid)
      }
      return Use-Rgb $hi[0] $hi[1] $hi[2] $solid
    }
    1 {
      $dark = Shift-Rgb $rgb -20
      if ($wideMode) {
        return (Use-Rgb $rgb[0] $rgb[1] $rgb[2] $solid) + (Use-Rgb $dark[0] $dark[1] $dark[2] $dense)
      }
      return Use-Rgb $dark[0] $dark[1] $dark[2] $dense
    }
    2 {
      $soft = Shift-Rgb $rgb -10
      if ($wideMode) {
        return (Use-Rgb $soft[0] $soft[1] $soft[2] $medium) + (Use-Rgb $rgb[0] $rgb[1] $rgb[2] $solid)
      }
      return Use-Rgb $soft[0] $soft[1] $soft[2] $medium
    }
    3 {
      $dark = Shift-Rgb $rgb -24
      if ($wideMode) {
        return (Use-Rgb $dark[0] $dark[1] $dark[2] $dense) + (Use-Rgb $rgb[0] $rgb[1] $rgb[2] $solid)
      }
      return Use-Rgb $rgb[0] $rgb[1] $rgb[2] $solid
    }
    4 {
      $hi = Shift-Rgb $rgb 8
      if ($wideMode) {
        return (Use-Rgb $hi[0] $hi[1] $hi[2] $solid) + (Use-Rgb $rgb[0] $rgb[1] $rgb[2] $medium)
      }
      return Use-Rgb $hi[0] $hi[1] $hi[2] $solid
    }
    default {
      $deep = Shift-Rgb $rgb -30
      if ($wideMode) {
        return (Use-Rgb $rgb[0] $rgb[1] $rgb[2] $solid) + (Use-Rgb $deep[0] $deep[1] $deep[2] $light)
      }
      return Use-Rgb $deep[0] $deep[1] $deep[2] $medium
    }
  }
}

function Render-TexturedShadowCell($rgb, [int]$row, [int]$col, [bool]$wideMode) {
  $dense = [string](U "2593")
  $medium = [string](U "2592")
  $light = [string](U "2591")
  $pattern = (($row * 3) + $col) % 3
  if ($wideMode) {
    if ($pattern -eq 0) {
      return (Use-Rgb $rgb[0] $rgb[1] $rgb[2] $dense) + (Use-Rgb $rgb[0] $rgb[1] $rgb[2] $medium)
    }
    if ($pattern -eq 1) {
      return (Use-Rgb $rgb[0] $rgb[1] $rgb[2] $medium) + (Use-Rgb $rgb[0] $rgb[1] $rgb[2] $light)
    }
    return (Use-Rgb $rgb[0] $rgb[1] $rgb[2] $dense) + (Use-Rgb $rgb[0] $rgb[1] $rgb[2] $dense)
  }

  if ($pattern -eq 1) {
    return Use-Rgb $rgb[0] $rgb[1] $rgb[2] $medium
  }
  return Use-Rgb $rgb[0] $rgb[1] $rgb[2] $dense
}

function Get-FemeciBrandRgb([int]$letterIndex, [int]$row, [int]$col) {
  $magenta = @(190, 18, 137)
  $deepMagenta = @(156, 21, 124)
  $cyan = @(24, 190, 202)
  $aqua = @(80, 215, 214)
  $blue = @(0, 83, 166)
  $yellow = @(255, 194, 14)
  $green = @(0, 172, 95)
  $purple = @(92, 20, 134)
  $orange = @(247, 148, 29)

  switch ($letterIndex) {
    0 {
      if ($row -ge 4) { return $yellow }
      return $magenta
    }
    1 {
      if ($row -ge 4) { return $blue }
      if ($row -eq 3) { return $aqua }
      return $cyan
    }
    2 {
      if ($row -le 1) {
        if (($col % 4) -lt 2) { return $purple }
        return $yellow
      }
      if ($row -le 3) {
        if (($col % 5) -lt 2) { return $deepMagenta }
        return $cyan
      }
      if (($col % 6) -lt 3) { return $green }
      return $aqua
    }
    3 {
      if ($row -le 2) { return $cyan }
      if ($row -eq 3) { return $blue }
      if ($row -eq 4) { return $deepMagenta }
      return $cyan
    }
    4 {
      if ($row -le 1) {
        if (($col % 4) -lt 2) { return $cyan }
        return $deepMagenta
      }
      if ($row -ge 5) { return $yellow }
      return $aqua
    }
    default {
      if ($row -le 1) { return $blue }
      if ($row -ge 4) { return $deepMagenta }
      return $orange
    }
  }
}

function Draw-Wordmark {
  $grid = New-FemeciBlockGrid
  $filled = $grid.Filled
  $letters = $grid.Letters
  $shadow = $grid.Shadow
  $shadowLetters = $grid.ShadowLetters
  $wideMode = $true
  $blank = "  "
  $logoWidth = $grid.Width * 2
  $terminalWidth = Get-TerminalWidth
  if ($terminalWidth -lt ($logoWidth + $script:PanelLeft + 2)) {
    $wideMode = $false
    $blank = " "
    $logoWidth = $grid.Width
  }
  $shadowColors = @(
    @(78, 26, 88),
    @(12, 48, 87),
    @(42, 31, 92),
    @(8, 85, 92),
    @(88, 64, 18),
    @(72, 33, 75)
  )

  for ($row = 0; $row -lt $grid.Height; $row++) {
    $line = ""
    for ($col = 0; $col -lt $grid.Width; $col++) {
      $key = "$row,$col"
      if (Test-FemeciCell $filled $row $col) {
        $letterIndex = $letters["$row,$col"]
        $rgb = Get-FemeciBrandRgb $letterIndex $row $col
        $topEdge = -not (Test-FemeciCell $filled ($row - 1) $col)
        $leftEdge = -not (Test-FemeciCell $filled $row ($col - 1))
        $rightEdge = -not (Test-FemeciCell $filled $row ($col + 1))
        $bottomEdge = -not (Test-FemeciCell $filled ($row + 1) $col)
        $line += Render-TexturedFrontCell $rgb $row $col $wideMode $topEdge $leftEdge $rightEdge $bottomEdge
      } elseif ($shadow.ContainsKey($key)) {
        $letterIndex = $shadowLetters[$key]
        $rgb = $shadowColors[[Math]::Min([Math]::Max($letterIndex, 0), $shadowColors.Count - 1)]
        $line += Render-TexturedShadowCell $rgb $row $col $wideMode
      } else {
        $line += $blank
      }
    }
    if ($line.Trim()) {
      Write-Line ((Get-Indent $logoWidth) + $line.TrimEnd())
    }
  }

  Write-Line ""
  $meta = "Proyecto de " + $script:Senias + "  |  Feria Mexicana de Ciencias e " + $script:Ingenierias
  $coloredMeta = (Use-Rgb 190 18 137 ("Proyecto de " + $script:Senias)) + (Use-Color "90" "  |  ") + (Use-Color "97" ("Feria Mexicana de Ciencias e " + $script:Ingenierias))
  Write-LogoLine $coloredMeta $meta

  $project = "Traductor de " + $script:Senias + " IA"
  $coloredProject = Use-Rgb 24 190 202 ("Traductor de " + $script:Senias + " IA")
  Write-LogoLine $coloredProject $project
}

function Write-InstitutionalRail {
  if ((Get-TerminalWidth) -lt 104) {
    $plain1 = "COAHUILA PA' DELANTE  |  A PASOS DE GIGANTE"
    $plain2 = "IMPULSO " + $script:Educacion + "  |  SEDU " + $script:Secretaria + " de " + $script:Educacion
    $plain3 = "COECYT Ciencia y " + $script:Tecnologia + "  |  Desarrollo Humano Coahuila"
    $line1 = (Use-Color "97" "COAHUILA") + (Use-Rgb 24 190 202 " PA' DELANTE") + (Use-Color "90" "  |  ") + (Use-Rgb 255 194 14 "A PASOS DE GIGANTE")
    $line2 = (Use-Rgb 190 18 137 "IMPULSO") + (Use-Color "97" (" " + $script:Educacion)) + (Use-Color "90" "  |  ") + (Use-Color "97;45" (" SEDU " + $script:Secretaria + " de " + $script:Educacion + " "))
    $line3 = (Use-Color "97;46" (" COECYT Ciencia y " + $script:Tecnologia + " ")) + (Use-Color "90" "  |  ") + (Use-Rgb 190 18 137 "Desarrollo Humano Coahuila")
    Write-LogoLine $line1 $plain1
    Write-LogoLine $line2 $plain2
    Write-LogoLine $line3 $plain3
  } else {
    $plain1 = "COAHUILA PA' DELANTE  |  A PASOS DE GIGANTE  |  IMPULSO " + $script:Educacion
    $plain2 = "SEDU " + $script:Secretaria + " de " + $script:Educacion + "  |  COECYT Ciencia y " + $script:Tecnologia + "  |  Desarrollo Humano Coahuila"
    $line1 = (Use-Color "97" "COAHUILA") + (Use-Rgb 24 190 202 " PA' DELANTE") + (Use-Color "90" "  |  ") + (Use-Rgb 255 194 14 "A PASOS DE GIGANTE") + (Use-Color "90" "  |  ") + (Use-Rgb 190 18 137 "IMPULSO") + (Use-Color "97" (" " + $script:Educacion))
    $line2 = (Use-Color "97;45" (" SEDU " + $script:Secretaria + " de " + $script:Educacion + " ")) + (Use-Color "90" "  |  ") + (Use-Color "97;46" (" COECYT Ciencia y " + $script:Tecnologia + " ")) + (Use-Color "90" "  |  ") + (Use-Rgb 190 18 137 "Desarrollo Humano Coahuila")
    Write-LogoLine $line1 $plain1
    Write-LogoLine $line2 $plain2
  }
}

function Get-ProjectTone {
  if (Test-Path $script:ModelQuality) {
    return "modelo disponible con reporte de calidad"
  }
  if (Test-Path $script:ModelKeras) {
    return "modelo disponible"
  }
  return "modelo pendiente de validar"
}

function Draw-Header($section = "") {
  Clear-Panel
  for ($i = 0; $i -lt (Get-TopPadding); $i++) {
    Write-Line ""
  }
  Draw-Wordmark
  Write-Line ""
  Write-InstitutionalRail
  Write-Line ""
}

function Draw-Footer {
  if (-not (Use-CompactVerticalLayout)) {
    Write-Line ""
  }
  Write-PanelLine ((Use-Color "90" "Flechas: seleccionar") + "     " + (Use-Color "90" "Enter: abrir") + "     " + (Use-Color "90" "Esc: salir") + "     " + (Use-Color "90" "O: carpeta"))
}

function Fit-Text([string]$text, [int]$width) {
  if ($null -eq $text) {
    $text = ""
  }
  if ($width -le 0) {
    return ""
  }
  if ($text.Length -gt $width) {
    if ($width -eq 1) {
      return "."
    }
    return $text.Substring(0, $width - 1) + "."
  }
  return $text + (" " * ($width - $text.Length))
}

function Get-MenuTableLayout {
  $terminalWidth = Get-TerminalWidth
  $tableWidth = [Math]::Min(78, [Math]::Max(52, $terminalWidth - 8))
  $innerWidth = $tableWidth - 2
  $labelWidth = 25
  $hintWidth = [Math]::Max(16, $innerWidth - 36)
  return @{
    TableWidth = $tableWidth
    InnerWidth = $innerWidth
    LabelWidth = $labelWidth
    HintWidth = $hintWidth
    H = [string](U "2500")
    V = [string](U "2502")
    TL = [string](U "250C")
    TR = [string](U "2510")
    BL = [string](U "2514")
    BR = [string](U "2518")
    LT = [string](U "251C")
    RT = [string](U "2524")
  }
}

function Format-MenuRow($item, [int]$index, [int]$selected, $layout) {
  $indexText = "{0:00}" -f ($index + 1)
  $marker = " "
  if ($index -eq $selected) {
    $marker = ">"
  }
  $label = Fit-Text $item.Label $layout.LabelWidth
  $hint = Fit-Text $item.Hint $layout.HintWidth
  $row = " $marker $indexText  $label  $hint "
  $row = Fit-Text $row $layout.InnerWidth
  if ($index -eq $selected) {
    return $layout.V + (Use-Color "30;46" $row) + $layout.V
  }
  $split = [Math]::Min(34, $row.Length)
  return $layout.V + (Use-Color "97" $row.Substring(0, $split)) + (Use-Color "90" $row.Substring($split)) + $layout.V
}

function Write-MenuTable($title, $subtitle, $items, [int]$selected) {
  $layout = Get-MenuTableLayout
  $rowTops = @()

  if ($title) {
    Write-PanelLine (Use-Rgb 24 190 202 $title) $layout.TableWidth
  }
  if ($subtitle) {
    Write-PanelLine (Use-Color "90" $subtitle) $layout.TableWidth
  }
  if ($title -or $subtitle) {
    Write-Line ""
  }

  Write-PanelLine (($layout.TL + ($layout.H * $layout.InnerWidth) + $layout.TR)) $layout.TableWidth
  for ($i = 0; $i -lt $items.Count; $i++) {
    $rowTops += (Get-CursorTopSafe)
    Write-PanelLine (Format-MenuRow $items[$i] $i $selected $layout) $layout.TableWidth
    if ($i -lt ($items.Count - 1)) {
      Write-PanelLine (($layout.LT + ($layout.H * $layout.InnerWidth) + $layout.RT)) $layout.TableWidth
    }
  }
  Write-PanelLine (($layout.BL + ($layout.H * $layout.InnerWidth) + $layout.BR)) $layout.TableWidth
  return @{
    Layout = $layout
    RowTops = $rowTops
    Width = (Get-TerminalWidth)
    Height = (Get-TerminalHeight)
  }
}

function Update-MenuRow($state, $items, [int]$index, [int]$selected) {
  if ($null -eq $state -or $null -eq $state.RowTops -or $index -ge $state.RowTops.Count) {
    return $false
  }
  $top = [int]$state.RowTops[$index]
  if ($top -lt 0) {
    return $false
  }
  if (-not (Set-CursorSafe 0 $top)) {
    return $false
  }
  Write-Host -NoNewline "$script:Esc[2K"
  Write-Host -NoNewline ((Get-Indent $state.Layout.TableWidth) + (Format-MenuRow $items[$index] $index $selected $state.Layout))
  return $true
}

function Can-UseFastMenuUpdate($state) {
  if ($null -eq $state) {
    return $false
  }
  return ((Get-TerminalWidth) -eq $state.Width -and (Get-TerminalHeight) -eq $state.Height)
}

function Invoke-Batch($fileName) {
  $path = Join-Path $script:Root $fileName
  if (-not (Test-Path $path)) {
    Draw-Header "Archivo no encontrado"
    Write-Line ("     " + (Use-Color "91" $fileName))
    Wait-Key
    return
  }

  Clear-Panel
  Write-Line ""
  Write-Line ("     " + (Use-Color "96" ("Ejecutando " + $fileName)))
  Write-Line ("     " + (Use-Color "90" $path))
  Write-Line ""
  & $path
  Wait-Key
}

function Need-Python {
  if (Test-Path $script:VenvPython) {
    return $true
  }

  Draw-Header "Entorno pendiente"
  Write-Line ("     " + (Use-Color "91" "No se encontro el python externo del proyecto."))
  Write-Line ("     " + (Use-Color "90" $script:VenvPython))
  Write-Line ""
  Write-Line ("     Usa " + (Use-Color "97" "Herramientas Tecnicas > Reparar entorno") + ".")
  Wait-Key
  return $false
}

function Invoke-PythonArgs([string[]]$arguments) {
  if (-not (Need-Python)) {
    return
  }

  Clear-Panel
  Write-Line ""
  Write-Line ("     " + (Use-Color "96" ("Ejecutando " + ($arguments -join " "))))
  Write-Line ""
  Push-Location $script:Codigo
  try {
    & $script:VenvPython @arguments
  } finally {
    Pop-Location
  }
  Wait-Key
}

function Invoke-Translator([string[]]$extraArgs) {
  if (-not (Need-Python)) {
    return
  }
  if (-not (Test-Path $script:TranslatorScript)) {
    Draw-Header ("Aplicaci" + $script:Oo + "n no encontrada")
    Write-Line ("     " + (Use-Color "91" $script:TranslatorScript))
    Wait-Key
    return
  }

  $arguments = @($script:TranslatorScript) + $extraArgs
  Invoke-PythonArgs $arguments
}

function Invoke-VerifyProject {
  if (-not (Need-Python)) {
    return
  }

  Clear-Panel
  Write-Line ""
  Write-Line ("     " + (Use-Color "96" "Verificando proyecto reestructurado"))
  Write-Line ""

  Push-Location $script:Codigo
  try {
    & $script:VenvPython -m compileall -q "aplicacion" "entrenamiento" "herramientas" "traductor_senas"
    if ($LASTEXITCODE -ne 0) {
      throw "compileall fallo"
    }
    & $script:VenvPython $script:PrepareScript "--check"
    if ($LASTEXITCODE -ne 0) {
      throw "revision de entorno fallo"
    }
    & $script:VenvPython $script:TranslatorScript "--camera" "0" "--smoke-test"
    if ($LASTEXITCODE -ne 0) {
      throw "prueba rapida del traductor fallo"
    }
  } catch {
    Write-Line ""
    Write-Line ("     " + (Use-Color "91" ("Error: " + $_.Exception.Message)))
  } finally {
    Pop-Location
  }

  Wait-Key
}

function Open-Path($path) {
  if (Test-Path $path) {
    Start-Process $path
  } else {
    Draw-Header "No encontrado"
    Write-Line ("     " + (Use-Color "91" $path))
    Wait-Key
  }
}

function Show-Information {
  Draw-Header
  Write-PanelLine (Use-Rgb 24 190 202 $script:Informacion)
  Write-Line ""
  Write-PanelLine ((Use-Color "97" "Proyecto") + (Use-Color "90" "      Traductor de " + $script:Senias + " IA"))
  Write-PanelLine ((Use-Color "97" "Area") + (Use-Color "90" "          Vision por computadora aplicada a " + $script:Senias + "."))
  Write-PanelLine ((Use-Color "97" "Tecnologia") + (Use-Color "90" "    Python, OpenCV, MediaPipe y TensorFlow/Keras."))
  Write-PanelLine ((Use-Color "97" "Estado") + (Use-Color "90" "        " + (Get-ProjectTone)))
  Write-Line ""
  Write-PanelLine ((Use-Color "97" "Python") + (Use-Color "90" "       ") + (Status-Text $script:VenvPython))
  Write-PanelLine ((Use-Color "97" ("Aplicaci" + $script:Oo + "n")) + (Use-Color "90" "   ") + (Status-Text $script:TranslatorScript))
  Write-PanelLine ((Use-Color "97" "Modelo Keras") + (Use-Color "90" "  ") + (Status-Text $script:ModelKeras))
  Write-PanelLine ((Use-Color "97" "Modelo TFLite") + (Use-Color "90" " ") + (Status-Text $script:ModelTflite))
  Write-PanelLine ((Use-Color "97" ("Im" + $script:Aa + "genes")) + (Use-Color "90" "     ") + (Status-Text $script:Images))
  Write-PanelLine ((Use-Color "97" "Datos") + (Use-Color "90" "        ") + (Status-Text $script:ProcessedData))
  Write-PanelLine ((Use-Color "97" "Reportes") + (Use-Color "90" "     ") + (Status-Text $script:Reports))
  Wait-Key
}

function Show-Authors {
  Draw-Header
  Write-PanelLine (Use-Rgb 24 190 202 ("Autores y " + $script:Creditos))
  Write-Line ""
  Write-PanelLine ((Use-Color "97" "Proyecto") + (Use-Color "90" "      Traductor de " + $script:Senias + " IA"))
  Write-PanelLine ((Use-Color "97" "Autor") + (Use-Color "90" "         Angel Zamora"))
  Write-PanelLine ((Use-Color "97" "Marco") + (Use-Color "90" "         Feria Mexicana de Ciencias e " + $script:Ingenierias))
  Write-Line ""
  Write-PanelLine (Use-Color "97" "Atribuciones")
  Write-PanelLine (Use-Color "90" "Referencia visual de dactilologia: Wikimedia Commons.")
  Write-PanelLine (Use-Color "90" "File: Deletreo de LSM.png, autora GUADALUPE GOMEZ TAPIA.")
  Write-PanelLine (Use-Color "90" "Licencia Creative Commons Attribution-ShareAlike 4.0.")
  Wait-Key
}

function Show-Menu($title, $subtitle, $items) {
  $selected = 0
  while ($true) {
    Draw-Header
    $menuState = Write-MenuTable $title $subtitle $items $selected
    Draw-Footer

    $redraw = $false
    while (-not $redraw) {
      $key = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
      switch ($key.VirtualKeyCode) {
        27 { return }
        38 {
          $previous = $selected
          $selected = ($selected - 1 + $items.Count) % $items.Count
          if ((Can-UseFastMenuUpdate $menuState) -and (Update-MenuRow $menuState $items $previous $selected) -and (Update-MenuRow $menuState $items $selected $selected)) {
            continue
          }
          $redraw = $true
        }
        40 {
          $previous = $selected
          $selected = ($selected + 1) % $items.Count
          if ((Can-UseFastMenuUpdate $menuState) -and (Update-MenuRow $menuState $items $previous $selected) -and (Update-MenuRow $menuState $items $selected $selected)) {
            continue
          }
          $redraw = $true
        }
        13 {
          $result = & $items[$selected].Action
          if ($result -eq "back") { return }
          if ($result -eq "exit") { return "exit" }
          $redraw = $true
        }
        79 {
          Open-Path $script:Root
          $redraw = $true
        }
      }
    }
  }
}

function Show-UseModel {
  Invoke-Translator @("--camera", "0", "--recognition-mode", "auto", "--stability-window", "7", "--stability-votes", "4")
}

function Show-WebInterface {
  if (-not (Need-Python)) {
    return
  }
  if (-not (Test-Path $script:InterfaceDist)) {
    Draw-Header "Interfaz pendiente"
    Write-Line ("     " + (Use-Color "91" "Todavia falta construir la interfaz ui."))
    Write-Line ("     " + (Use-Color "90" $script:InterfaceDist))
    Write-Line ""
    Write-Line ("     Ejecuta en codigo\interfaz ui: " + (Use-Color "97" "npm install") + " y " + (Use-Color "97" "npm run build") + ".")
    Wait-Key
    return
  }
  $url = "http://127.0.0.1:8765"
  try {
    $response = Invoke-WebRequest -Uri ($url + "/api/estado") -UseBasicParsing -TimeoutSec 1
    if ($response.StatusCode -eq 200) {
      Start-Process $url
      return
    }
  } catch {
  }
  $arguments = "-NoProfile -ExecutionPolicy Bypass -Command `"& '$script:VenvPython' '$script:InterfaceServer'`""
  Start-Process -FilePath $script:PowerShellExe -ArgumentList $arguments -WindowStyle Minimized
}

function Show-Folders {
  $items = @(
    @{ Label = "Proyecto"; Hint = $script:Root; Action = { Open-Path $script:Root } },
    @{ Label = "Codigo"; Hint = $script:Codigo; Action = { Open-Path $script:Codigo } },
    @{ Label = "Im" + $script:Aa + "genes de entrenamiento"; Hint = $script:Images; Action = { Open-Path $script:Images } },
    @{ Label = "Datos procesados"; Hint = $script:ProcessedData; Action = { Open-Path $script:ProcessedData } },
    @{ Label = "Reportes"; Hint = $script:Reports; Action = { Open-Path $script:Reports } },
    @{ Label = "Archivos externos"; Hint = $script:ExternalBase; Action = { Open-Path $script:ExternalBase } },
    @{ Label = "Pruebas externas"; Hint = $script:TestsBase; Action = { Open-Path $script:TestsBase } },
    @{ Label = "Volver"; Hint = ""; Action = { return "back" } }
  )
  Show-Menu "Abrir carpetas" "" $items
}

function Show-References {
  $items = @(
    @{ Label = "Calidad del modelo"; Hint = "model_quality.json"; Action = { Open-Path $script:ModelQuality } },
    @{ Label = "Requisitos"; Hint = "dependencias usadas"; Action = { Open-Path (Join-Path $script:Codigo "herramientas\requisitos.txt") } },
    @{ Label = "Argumentos"; Hint = "par" + $script:Aa + "metros disponibles"; Action = { Open-Path (Join-Path $script:Codigo "aplicacion\argumentos.py") } },
    @{ Label = "Im" + $script:Aa + "genes de entrenamiento"; Hint = "dataset visual"; Action = { Open-Path $script:Images } },
    @{ Label = "Datos procesados"; Hint = "entrenamiento y salidas"; Action = { Open-Path $script:ProcessedData } },
    @{ Label = "Abrir carpeta de reportes"; Hint = $script:Reports; Action = { Open-Path $script:Reports } },
    @{ Label = "Volver"; Hint = ""; Action = { return "back" } }
  )
  Show-Menu "Referencias" "" $items
}

function Show-Tools {
  $items = @(
    @{ Label = "Revisar entorno"; Hint = "dependencias y rutas"; Action = { Invoke-PythonArgs @($script:PrepareScript, "--check") } },
    @{ Label = "Reparar entorno"; Hint = "instala faltantes"; Action = { Invoke-PythonArgs @($script:PrepareScript, "--repair") } },
    @{ Label = "Verificar proyecto"; Hint = "compila y carga modelo"; Action = { Invoke-VerifyProject } },
    @{ Label = "Verificar flujo web"; Hint = "interfaz, api y modelo"; Action = { Invoke-PythonArgs @("herramientas\verificar_flujo_web.py") } },
    @{ Label = "Verificar interfaz"; Hint = "limpieza visual y texto real"; Action = { Invoke-PythonArgs @("herramientas\verificar_interfaz.py") } },
    @{ Label = "Estado real del entrenamiento"; Hint = "catalogo, secuencias y modelo"; Action = { Invoke-PythonArgs @("herramientas\revisar_estado_entrenamiento.py") } },
    @{ Label = "Auditar datos"; Hint = "duplicados y faltantes"; Action = { Invoke-PythonArgs @("herramientas\auditar_datos_entrenamiento.py") } },
    @{ Label = "Entrenar modelo"; Hint = "entrenamiento local"; Action = { Invoke-PythonArgs @("entrenamiento\entrenador.py") } },
    @{ Label = "Importar im" + $script:Aa + "genes LSM"; Hint = "ver par" + $script:Aa + "metros"; Action = { Invoke-PythonArgs @("entrenamiento\importar_imagenes_lsm.py", "--help") } },
    @{ Label = "Abrir c" + $script:Oo + "digo"; Hint = $script:Codigo; Action = { Open-Path $script:Codigo } },
    @{ Label = "Abrir reportes"; Hint = $script:Reports; Action = { Open-Path $script:Reports } },
    @{ Label = "Volver"; Hint = ""; Action = { return "back" } }
  )
  Show-Menu ("Herramientas " + $script:Tecnicas) "" $items
}

function Run-Main {
  $items = @(
    @{ Label = "Abrir interfaz"; Hint = "UI moderna del traductor"; Action = { Show-WebInterface } },
    @{ Label = "Abrir traductor directo"; Hint = $script:Camara + " y modelo en vivo"; Action = { Show-UseModel } },
    @{ Label = $script:Informacion; Hint = "Datos generales del proyecto"; Action = { Show-Information } },
    @{ Label = "Abrir carpetas"; Hint = "Proyecto, c" + $script:Oo + "digo y datos"; Action = { Show-Folders } },
    @{ Label = "Referencias"; Hint = "Gu" + $script:Ii + "as y fuentes"; Action = { Show-References } },
    @{ Label = "Autores y " + $script:Creditos; Hint = "Equipo y atribuciones"; Action = { Show-Authors } },
    @{ Label = "Herramientas " + $script:Tecnicas; Hint = "Configuraci" + $script:Oo + "n y reportes"; Action = { Show-Tools } },
    @{ Label = "Salir"; Hint = "Cerrar el panel."; Action = { return "exit" } }
  )

  [Console]::CursorVisible = $false
  try {
    $result = Show-Menu "" "" $items
    if ($result -eq "exit") {
      return
    }
  } finally {
    [Console]::CursorVisible = $true
    Write-Host -NoNewline "$script:Esc[0m"
    Clear-Panel
    Write-Line (Use-Color "96" "Panel cerrado.")
  }
}

if ($Preview) {
  Draw-Header
  $previewItems = @(
    @{ Label = "Abrir interfaz"; Hint = "UI moderna del traductor" },
    @{ Label = "Abrir traductor directo"; Hint = $script:Camara + " y modelo en vivo" },
    @{ Label = $script:Informacion; Hint = "Datos generales del proyecto" },
    @{ Label = "Abrir carpetas"; Hint = "Proyecto, c" + $script:Oo + "digo y datos" },
    @{ Label = "Referencias"; Hint = "Gu" + $script:Ii + "as y fuentes" },
    @{ Label = "Autores y " + $script:Creditos; Hint = "Equipo y atribuciones" },
    @{ Label = "Herramientas " + $script:Tecnicas; Hint = "Configuraci" + $script:Oo + "n y reportes" },
    @{ Label = "Salir"; Hint = "Cerrar el panel." }
  )
  $null = Write-MenuTable "" "" $previewItems 0
  Draw-Footer
  Write-Line ""
  Write-PanelLine (Use-Color "92" "Vista previa OK. Ejecuta abrir traductor.bat para usarlo.")
  exit 0
}

if ($Web) {
  Show-WebInterface
  exit 0
}

Run-Main

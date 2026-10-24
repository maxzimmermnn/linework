#!/bin/bash
# Builds dist/Linework.app from this checkout. Move it to /Applications and open it.
# Rebuild after pulling new code. Needs nothing but macOS; the app sets up its own
# Python environment on first start.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/dist/Linework.app"
VERSION="${VERSION:-1.0.$(git -C "$ROOT" rev-list --count HEAD 2>/dev/null || echo 0)}"  # release builds: VERSION=1.2.0

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/src"
rsync -a --exclude __pycache__ --exclude .DS_Store --exclude AppIcon.icns \
  "$ROOT/app.py" "$ROOT/requirements.txt" "$ROOT/linework" "$ROOT/macos" \
  "$ROOT/LICENSE" "$ROOT/THIRD_PARTY_NOTICES.md" "$ROOT/licenses" "$APP/Contents/Resources/src/"
cp "$ROOT/macos/launcher.sh" "$APP/Contents/MacOS/Linework"
chmod +x "$APP/Contents/MacOS/Linework"

cp "$ROOT/macos/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"  # redraw with macos/make_icon.py

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Linework</string>
  <key>CFBundleDisplayName</key><string>Linework</string>
  <key>CFBundleIdentifier</key><string>io.github.maxzimmermnn.linework</string>
  <key>CFBundleExecutable</key><string>Linework</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

codesign --force --sign - "$APP" 2>/dev/null || true  # ad-hoc signature, no Apple account needed
echo "Built $APP ($VERSION)"

# Reuse models already downloaded for this checkout, so the app does not fetch them again.
SUPPORT="$HOME/Library/Application Support/Linework"
if [ -d "$ROOT/weights" ] && [ ! -d "$SUPPORT/weights" ]; then
  mkdir -p "$SUPPORT"
  cp -R "$ROOT/weights" "$SUPPORT/weights"
  echo "Copied downloaded models to $SUPPORT/weights"
fi

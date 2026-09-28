// `npx remotion studio` / `npx remotion render` 用の設定（scripts/render.mjs も同じ値を使う）
import {Config} from '@remotion/cli/config';
import fs from 'node:fs';

const chrome = [
	process.env.REMOTION_CHROME,
	'/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell',
	'/opt/pw-browsers/chromium',
].find((p) => p && fs.existsSync(p));
if (chrome) {
	Config.setBrowserExecutable(chrome);
}
Config.setEntryPoint('src/index.ts');
Config.setPublicDir('public');
Config.setCodec('h264');
Config.setCrf(17);
Config.setPixelFormat('yuv420p');
Config.setColorSpace('bt709');
Config.setVideoImageFormat('png');
Config.setX264Preset('slow');
Config.setChromiumOpenGlRenderer('swangle');
Config.setOverwriteOutput(true);

#!/usr/bin/env node
// 6本（または指定した本）をまとめて書き出す。
//
//   node scripts/render.mjs                       # 6本すべて → ../exports/final/
//   node scripts/render.mjs IG30 IG15             # 指定した Composition だけ
//   node scripts/render.mjs IG30 --props bgm.json # BGMなどの入力（props）を渡す
//   node scripts/render.mjs Signage30 --out out --frames 0-59   # 試し書き（remotion/out/ へ）
//
// 仕様：H.264 High・CRF 17・yuv420p・BT.709（タグ付き・limited）。
//       サイネージは音声トラックなし（muted）。IGは AAC 48kHz 256kbps。
//
// IG の音について：Remotion が直接 mp4 に AAC で書くと、音が映像より 2048 サンプル（約43ms）遅れる
// （AAC の頭の遅れが補正されない。実測）。そこで IG は、いったん h264-mkv（映像＋無圧縮PCM、ずれなし）で
// 書き出し、Remotion 同梱の ffmpeg で映像はそのまま・音だけ AAC にして mp4 に詰め替える（ずれ0を確認済み）。
import {bundle} from '@remotion/bundler';
import {renderMedia, selectComposition, ensureBrowser} from '@remotion/renderer';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const RM = path.dirname(HERE);
const ROOT = path.dirname(RM);

// 作業環境にあるChromium（Remotionの自動ダウンロードは使わない）。REMOTION_CHROME で上書きできる
const CHROME_CANDIDATES = [
	process.env.REMOTION_CHROME,
	'/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell',
	'/opt/pw-browsers/chromium',
].filter(Boolean);
const browserExecutable = CHROME_CANDIDATES.find((p) => fs.existsSync(p)) ?? null;

const args = process.argv.slice(2);
const opt = (name, def) => {
	const i = args.indexOf(name);
	if (i < 0) return def;
	const v = args[i + 1];
	args.splice(i, 2);
	return v;
};
const outDir = path.resolve(RM, opt('--out', path.join(ROOT, 'exports', 'final')));
const propsFile = opt('--props', null);
const framesArg = opt('--frames', null);
const crf = Number(opt('--crf', '17'));
const concurrency = Number(opt('--concurrency', String(os.cpus().length)));

const timeline = JSON.parse(fs.readFileSync(path.join(RM, 'src', 'timeline.json'), 'utf8'));
const all = timeline.compositions.map((c) => c.id);
const ids = args.length ? args : all;
for (const id of ids) {
	if (!all.includes(id)) {
		console.error(`知らない Composition です：${id}（${all.join(', ')}）`);
		process.exit(1);
	}
}
const inputProps = propsFile ? JSON.parse(fs.readFileSync(path.resolve(propsFile), 'utf8')) : {};
fs.mkdirSync(outDir, {recursive: true});

console.log(`Chromium: ${browserExecutable ?? '（Remotionの既定）'}`);
if (!browserExecutable) await ensureBrowser();

const t0 = Date.now();
const serveUrl = await bundle({
	entryPoint: path.join(RM, 'src', 'index.ts'),
	publicDir: path.join(RM, 'public'),
	symlinkPublicDir: true,
	onProgress: () => {},
});
console.log(`バンドル ${((Date.now() - t0) / 1000).toFixed(1)}秒`);

// Remotion 同梱の ffmpeg（@remotion/compositor-*）
const compDir = fs
	.readdirSync(path.join(RM, 'node_modules', '@remotion'))
	.filter((d) => d.startsWith('compositor-'))
	.map((d) => path.join(RM, 'node_modules', '@remotion', d))[0];
const remux = (mkv, mp4, seconds) => {
	execFileSync(
		path.join(compDir, 'ffmpeg'),
		['-v', 'error', '-y', '-i', mkv, '-map', '0:v:0', '-map', '0:a:0', '-c:v', 'copy',
			'-c:a', 'aac', '-b:a', '256k', '-ar', '48000', '-t', seconds.toFixed(3), '-movflags', '+faststart', mp4],
		{env: {...process.env, LD_LIBRARY_PATH: compDir}, stdio: 'inherit'},
	);
};

const results = [];
for (const id of ids) {
	const meta = timeline.compositions.find((c) => c.id === id);
	const composition = await selectComposition({serveUrl, id, inputProps, browserExecutable, logLevel: 'error'});
	const outputLocation = path.join(outDir, meta.output);
	const tmpMkv = path.join(RM, 'out', meta.output.replace(/\.mp4$/, '.tmp.mkv'));
	if (meta.hasAudio) fs.mkdirSync(path.dirname(tmpMkv), {recursive: true});
	let frameRange = null;
	if (framesArg) {
		const [a, b] = framesArg.split('-').map(Number);
		frameRange = b === undefined ? a : [a, b];
	}
	const t = Date.now();
	let last = -1;
	await renderMedia({
		serveUrl,
		composition,
		inputProps,
		codec: meta.hasAudio ? 'h264-mkv' : 'h264', // h264-mkv の音は PCM 16bit・48kHz
		crf,
		pixelFormat: 'yuv420p',
		colorSpace: 'bt709',
		imageFormat: 'png', // 画面の取り込みは可逆（JPEGにしない）
		x264Preset: 'slow',
		muted: !meta.hasAudio, // サイネージは音声トラックを作らない
		outputLocation: meta.hasAudio ? tmpMkv : outputLocation,
		logLevel: 'error',
		browserExecutable,
		concurrency,
		frameRange,
		overwrite: true,
		chromiumOptions: {gl: 'swangle'},
		onProgress: ({progress}) => {
			const p = Math.floor(progress * 10);
			if (p !== last) {
				last = p;
				process.stdout.write(`\r${id} ${p * 10}%   `);
			}
		},
	});
	if (meta.hasAudio) {
		const frames = Array.isArray(frameRange) ? frameRange[1] - frameRange[0] + 1 : frameRange === null ? meta.durationInFrames : 1;
		remux(tmpMkv, outputLocation, frames / meta.fps);
		fs.unlinkSync(tmpMkv);
	}
	const sec = (Date.now() - t) / 1000;
	const mb = fs.statSync(outputLocation).size / 1048576;
	console.log(`\r${id} → ${path.relative(ROOT, outputLocation)}  ${sec.toFixed(1)}秒  ${mb.toFixed(1)}MB`);
	results.push({id, file: path.relative(ROOT, outputLocation), seconds: +sec.toFixed(1), mb: +mb.toFixed(2)});
}
// 書き出しの記録（同じ本は上書き、ほかの本の記録は残す）
const logPath = path.join(outDir, 'render_log.json');
let log = {};
try {
	log = JSON.parse(fs.readFileSync(logPath, 'utf8'));
} catch {}
for (const r of results) {
	log[r.id] = {...r, crf, renderedAt: new Date().toISOString(), props: inputProps};
}
fs.writeFileSync(logPath, JSON.stringify(log, null, 1));

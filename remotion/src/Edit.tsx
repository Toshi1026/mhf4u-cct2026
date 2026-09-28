import React from 'react';
import {
	AbsoluteFill,
	Audio,
	Img,
	OffthreadVideo,
	Sequence,
	interpolate,
	staticFile,
	useCurrentFrame,
	useVideoConfig,
} from 'remotion';
import colorFix from './colorFix.json';
import timelineJson from './timeline.json';
import type {Bgm, EditProps, Overlay, Timeline} from './types';

const TIMELINE = timelineJson as unknown as Timeline;

// OffthreadVideo の YUV→RGB（swscale の速い経路）の切り捨てを打ち消す 8bit の表（scripts/calibrate_color.py）
// Chrome は表の値をコマの8bitに戻すとき切り捨てる所があるので、+0.4 段ずらして置く（丸めでも切り捨てでも同じ値になる）
const table = (v: number[]) => v.map((x) => (Math.min(255, x + 0.4) / 255).toFixed(6)).join(' ');
const ColorFixFilter: React.FC = () => (
	<svg width={0} height={0} style={{position: 'absolute'}} aria-hidden>
		<filter id="maze-yuvfix" colorInterpolationFilters="sRGB" x="0" y="0" width="1" height="1">
			<feComponentTransfer>
				<feFuncR type="table" tableValues={table(colorFix.r)} />
				<feFuncG type="table" tableValues={table(colorFix.g)} />
				<feFuncB type="table" tableValues={table(colorFix.b)} />
			</feComponentTransfer>
		</filter>
	</svg>
);

const src = (p: string) => (/^https?:\/\//.test(p) ? p : staticFile(p));
const dbToGain = (db: number) => Math.pow(10, db / 20);

// 透過PNG：不透明度はキーフレーム（全体のコマ番号）を直線でつなぐ。区間の外には置かない
const OverlayLayer: React.FC<{o: Overlay}> = ({o}) => {
	const frame = useCurrentFrame(); // Sequence の中なので、o.from からの相対
	const abs = frame + o.from;
	const ks = o.keyframes;
	const opacity =
		ks.length === 1
			? ks[0][1]
			: interpolate(
					abs,
					ks.map((k) => k[0]),
					ks.map((k) => k[1]),
					{extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
				);
	return (
		<AbsoluteFill style={{opacity}}>
			<Img src={src(o.src)} style={{width: '100%', height: '100%'}} />
		</AbsoluteFill>
	);
};

// BGM の音量（コマごと）。フェード・ダッキングは dB で足し合わせる
export const bgmVolumeAt = (bgm: Bgm, tS: number, endS: number): number => {
	const startAt = bgm.startAtS ?? 0;
	const local = tS - startAt;
	let g = dbToGain(bgm.gainDb ?? 0);
	if (bgm.fadeInS && bgm.fadeInS > 0) {
		g *= interpolate(local, [0, bgm.fadeInS], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
	}
	if (bgm.fadeOutS && bgm.fadeOutS > 0) {
		g *= interpolate(tS, [endS - bgm.fadeOutS, endS], [1, 0], {
			extrapolateLeft: 'clamp',
			extrapolateRight: 'clamp',
		});
	}
	for (const d of bgm.ducking ?? []) {
		const r = d.rampS ?? 0.3;
		const depth = interpolate(tS, [d.fromS, d.fromS + r, d.toS - r, d.toS], [0, 1, 1, 0], {
			extrapolateLeft: 'clamp',
			extrapolateRight: 'clamp',
		});
		g *= dbToGain(d.gainDb * depth);
	}
	return g;
};

const BgmLayer: React.FC<{bgm: Bgm}> = ({bgm}) => {
	const {fps, durationInFrames} = useVideoConfig();
	const startAtF = Math.round((bgm.startAtS ?? 0) * fps);
	const endS = bgm.endAtS ?? durationInFrames / fps;
	const endF = Math.min(durationInFrames, Math.round(endS * fps));
	return (
		<Sequence from={startAtF} durationInFrames={Math.max(1, endF - startAtF)} name="BGM">
			<Audio
				src={src(bgm.src)}
				startFrom={Math.round((bgm.startFromS ?? 0) * fps)}
				// volume はシーケンス内のコマ番号で呼ばれる
				volume={(f) => bgmVolumeAt(bgm, (f + startAtF) / fps, endS)}
			/>
		</Sequence>
	);
};

export const Edit: React.FC<EditProps> = ({bgm, colorFix: fix = true}) => {
	const {id} = useVideoConfig();
	const timeline = TIMELINE.compositions.find((c) => c.id === id);
	if (!timeline) {
		throw new Error(`timeline.json に ${id} がありません`);
	}
	return (
		<AbsoluteFill style={{backgroundColor: 'black'}}>
			{fix ? <ColorFixFilter /> : null}
			<AbsoluteFill style={fix ? {filter: 'url(#maze-yuvfix)'} : undefined}>
			{timeline.clips.map((c) => (
				<Sequence
					key={`clip-${c.cut}`}
					from={c.from}
					durationInFrames={c.durationInFrames}
					name={`カット${c.cut} ${c.placeholder ? '（仮スレート）' : c.label}`}
				>
					<OffthreadVideo src={src(c.src)} muted toneMapped={false} />
				</Sequence>
			))}
			</AbsoluteFill>
			{timeline.overlays.map((o) => (
				<Sequence key={`ov-${o.label}`} from={o.from} durationInFrames={o.to - o.from} name={o.label}>
					<OverlayLayer o={o} />
				</Sequence>
			))}
			{timeline.audio ? <Audio src={src(timeline.audio.src)} volume={timeline.audio.volume} /> : null}
			{bgm && timeline.hasAudio ? <BgmLayer bgm={bgm} /> : null}
		</AbsoluteFill>
	);
};

// src/timeline.json の形（scripts/build_timeline.py が作る）
export type Clip = {
	cut: number;
	src: string; // public/ からの相対パス
	file: string; // maze/ からの相対パス（確認用）
	from: number;
	durationInFrames: number;
	placeholder: boolean;
	overridden: boolean;
	label: string;
};

export type Overlay = {
	label: string;
	src: string;
	file: string;
	from: number; // この区間だけ置く（from ≦ コマ < to）
	to: number;
	keyframes: [number, number][]; // [コマ番号（全体の通し番号）, 不透明度0〜1]。直線補間
};

export type AmbienceAudio = {src: string; file: string; volume: number};

export type EditTimeline = {
	id: string;
	fps: number;
	width: number;
	height: number;
	durationInFrames: number;
	clips: Clip[];
	overlays: Overlay[];
	audio: AmbienceAudio | null;
	hasAudio: boolean;
	output: string;
	reference: string;
	placeholders: number[];
	version: string;
};

export type Timeline = {fps: number; compositions: EditTimeline[]};

// BGM（初期値は null＝なし）。秒はすべてタイムライン上の秒（startFromS だけは曲の中の秒）
export type BgmDucking = {
	fromS: number; // ここから下げ始める
	toS: number; // ここで元に戻り終わる
	gainDb: number; // 下げる量（例：-6）
	rampS?: number; // 下げる／戻すのにかける秒（既定 0.3）
};

export type Bgm = {
	src: string; // public/ からの相対パス（例：bgm/17543.mp3）か http(s) のURL
	startFromS?: number; // 曲の何秒目から使うか（既定 0）
	startAtS?: number; // タイムラインの何秒目で鳴り始めるか（既定 0）
	endAtS?: number; // タイムラインの何秒目で止めるか（既定：最後まで）
	gainDb?: number; // 全体の音量（既定 0 dB）
	fadeInS?: number; // フェードインの長さ（既定 0）
	fadeOutS?: number; // 終わりのフェードアウトの長さ（既定 0）
	ducking?: BgmDucking[];
};

export type EditProps = {
	bgm: Bgm | null;
	// OffthreadVideo の色ずれの打ち消し（既定 true）。src/colorFix.json を使う
	colorFix?: boolean;
};

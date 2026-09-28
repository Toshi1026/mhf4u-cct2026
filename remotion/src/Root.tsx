import React from 'react';
import {Composition} from 'remotion';
import {Edit} from './Edit';
import timelineJson from './timeline.json';
import type {EditProps, Timeline} from './types';

const timeline = timelineJson as unknown as Timeline;

// 1本の書き出し＝1つの Composition。長さ・サイズ・カット・PNG・音はすべて timeline.json から
export const RemotionRoot: React.FC = () => {
	return (
		<>
			{timeline.compositions.map((t) => (
				<Composition
					key={t.id}
					id={t.id}
					component={Edit}
					width={t.width}
					height={t.height}
					fps={t.fps}
					durationInFrames={t.durationInFrames}
					defaultProps={{bgm: null} satisfies EditProps}
				/>
			))}
		</>
	);
};

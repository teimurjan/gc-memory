import {Composition} from "remotion";
import {LetheDemo} from "./Composition";

const FPS = 30;
const INTRO_FRAMES = 60;
const FLOW_FRAMES = 480;
const OUTRO_FRAMES = 150;
const DURATION = INTRO_FRAMES + FLOW_FRAMES + OUTRO_FRAMES;

export const Root: React.FC = () => {
  return (
    <Composition
      id="lethe-demo"
      component={LetheDemo}
      durationInFrames={DURATION}
      fps={FPS}
      width={1280}
      height={720}
    />
  );
};

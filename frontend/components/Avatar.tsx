'use client';

/**
 * 3D avatar with viseme-driven lip-sync.
 *
 * Ported from ds-catalogue-bot's AvatarWithLipSync: same GLB rig
 * (Shayla_Changes(Visemes).glb), same morph-target names, same `working.glb`
 * animation clips, same camera and spotlight rig — so it looks identical.
 *
 * Changes made, and why:
 *  * The per-frame `console.log` calls are gone. They fired inside `useFrame`
 *    (60/sec) and cost more than the morph interpolation itself.
 *  * `lerpMorphTarget` resolves the meshes that own each morph target once and
 *    caches them, instead of `scene.traverse()`-ing the whole hierarchy for every
 *    target on every frame. With 9 visemes plus blinks and brows that was ~13
 *    full traversals per frame.
 *  * Mouth shape is held in a ref, not React state. The previous version called
 *    `setState` on every Rhubarb cue, re-rendering the tree dozens of times per
 *    second for a value only the render loop reads.
 */

import React, {
  Suspense,
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
} from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Environment, Html, OrbitControls, useAnimations, useGLTF } from '@react-three/drei';
import * as THREE from 'three';

const AVATAR_GLB = '/Shayla_Changes(Visemes).glb';
const ANIMATION_GLB = '/working.glb';

/** Rhubarb / Azure mouth-shape letter → this rig's morph target name. */
const VISEME_MAP: Record<string, string> = {
  A: 'V_p1', // p, b, m — closed
  B: 'EE',   // i, j, k
  C: 'V_i1', // e, l, t, d, n
  D: 'Ah',   // open a
  E: 'V_o1', // o
  F: 'V_u1', // u, w, r
  G: 'V_f1', // f, v
  H: 'V_s2', // s, z, sh
  X: 'Er',   // rest
};

const ALL_VISEMES = Object.values(VISEME_MAP);

export interface AvatarHandle {
  setMouthShape: (shape: string) => void;
  reset: () => void;
}

interface ModelProps {
  isSpeaking: boolean;
  mouthShapeRef: React.MutableRefObject<string>;
}

function AvatarModel({ isSpeaking, mouthShapeRef }: ModelProps) {
  const { nodes, materials, scene } = useGLTF(AVATAR_GLB);
  const { animations } = useGLTF(ANIMATION_GLB);
  const group = useRef<THREE.Group>(null);
  const { actions } = useAnimations(animations, group);

  const lastBlink = useRef(0);
  const nextBlinkDelay = useRef(3);

  // Resolve morph-target owners once. scene.traverse() per target per frame was
  // the single biggest cost in the original component.
  const morphIndex = useMemo(() => {
    const map = new Map<string, Array<{ mesh: THREE.Mesh; index: number }>>();
    scene.traverse((child) => {
      const mesh = child as THREE.Mesh & {
        isSkinnedMesh?: boolean;
        morphTargetDictionary?: Record<string, number>;
        morphTargetInfluences?: number[];
      };
      if (!mesh.isSkinnedMesh || !mesh.morphTargetDictionary || !mesh.morphTargetInfluences) return;
      Object.entries(mesh.morphTargetDictionary).forEach(([name, index]) => {
        if (!map.has(name)) map.set(name, []);
        map.get(name)!.push({ mesh: mesh as THREE.Mesh, index });
      });
    });
    return map;
  }, [scene]);

  const lerpMorph = useCallback(
    (target: string, value: number, speed = 0.15) => {
      const owners = morphIndex.get(target);
      if (!owners) return;
      for (const { mesh, index } of owners) {
        const influences = (mesh as unknown as { morphTargetInfluences: number[] })
          .morphTargetInfluences;
        influences[index] = THREE.MathUtils.lerp(influences[index], value, speed);
      }
    },
    [morphIndex]
  );

  useEffect(() => {
    const idle = actions['Talking'] || actions[Object.keys(actions)[0] ?? ''];
    idle?.reset().fadeIn(0.5).play();
    return () => {
      idle?.fadeOut(0.5);
    };
  }, [actions]);

  useFrame((state) => {
    const time = state.clock.elapsedTime;

    // ---------------------------------------------------------------- blinking
    if (time - lastBlink.current > nextBlinkDelay.current) {
      lastBlink.current = time;
      nextBlinkDelay.current = 2 + Math.random() * 4;
    }
    const sinceBlink = time - lastBlink.current;
    let blink = 0;
    if (sinceBlink < 0.1) blink = sinceBlink / 0.1;
    else if (sinceBlink < 0.2) blink = 1 - (sinceBlink - 0.1) / 0.1;
    lerpMorph('Eye_Blink_L', blink, 0.5);
    lerpMorph('Eye_Blink_R', blink, 0.5);

    // ---------------------------------------------------------------- lip sync
    const shape = mouthShapeRef.current;
    const activeMorph = isSpeaking && shape !== 'X' ? VISEME_MAP[shape] : null;

    if (activeMorph) {
      lerpMorph(activeMorph, 0.85, 0.45);
      for (const morph of ALL_VISEMES) {
        if (morph !== activeMorph) lerpMorph(morph, 0, 0.32);
      }
    } else {
      lerpMorph(VISEME_MAP.X, 0.2, 0.2);
      for (const morph of ALL_VISEMES) {
        if (morph !== VISEME_MAP.X) lerpMorph(morph, 0, 0.2);
      }
    }

    // ------------------------------------------------------------- expression
    const expressive = isSpeaking ? 1 : 0;
    lerpMorph('Mouth_Smile_L', 0.1 * expressive, 0.05);
    lerpMorph('Mouth_Smile_R', 0.1 * expressive, 0.05);
    lerpMorph('Brow_Raise_Outer_L', 0.15 * expressive, 0.05);
    lerpMorph('Brow_Raise_Outer_R', 0.15 * expressive, 0.05);

    // A slow idle sway keeps the model from reading as a frozen render.
    if (group.current) {
      group.current.rotation.y = Math.sin(time * 0.35) * 0.035;
    }
  });

  const n = nodes as Record<string, any>;
  const m = materials as Record<string, any>;

  return (
    <group ref={group} dispose={null} position={[0, -2.9, 0]} scale={[2.7, 2.7, 2.7]}>
      <mesh
        castShadow
        receiveShadow
        geometry={n.glb_bg_1?.geometry}
        material={m['glb_bg 1']}
        position={[0, 1.456, -0.197]}
        rotation={[1.523, 0, 0]}
      />
      <skinnedMesh
        geometry={n.Bang?.geometry}
        material={m['Hair_Transparency.003']}
        skeleton={n.Bang?.skeleton}
      />
      <skinnedMesh
        geometry={n.Fit_shirts?.geometry}
        material={m.Fit_shirts}
        skeleton={n.Fit_shirts?.skeleton}
      />
      <skinnedMesh
        geometry={n.Real_Hair?.geometry}
        material={m['Hair_Transparency.002']}
        skeleton={n.Real_Hair?.skeleton}
      />
      {n.CC_Base_BoneRoot && <primitive object={n.CC_Base_BoneRoot} />}
      {[
        ['CC_Base_Body_1', 'Std_Tongue'],
        ['CC_Base_Body_2', 'Std_Skin_Head'],
        ['CC_Base_Body_3', 'Std_Eyelash'],
        ['CC_Base_Body_4', 'Std_Upper_Teeth'],
        ['CC_Base_Body_5', 'Std_Lower_Teeth'],
        ['CC_Base_Body_6', 'Std_Eye_R'],
        ['CC_Base_Body_7', 'Std_Cornea_R'],
        ['CC_Base_Body_8', 'Std_Eye_L'],
        ['CC_Base_Body_9', 'Std_Cornea_L'],
        ['CC_Base_TearLine_1', 'Std_Tearline_R'],
        ['CC_Base_TearLine_2', 'Std_Tearline_L'],
      ].map(([nodeName, materialName]) => (
        <skinnedMesh
          key={nodeName}
          name={nodeName}
          geometry={n[nodeName]?.geometry}
          material={m[materialName]}
          skeleton={n[nodeName]?.skeleton}
          morphTargetDictionary={n[nodeName]?.morphTargetDictionary}
          morphTargetInfluences={n[nodeName]?.morphTargetInfluences}
        />
      ))}
      <skinnedMesh
        geometry={n.Hair_Base_1?.geometry}
        material={m.Hair_Transparency}
        skeleton={n.Hair_Base_1?.skeleton}
      />
      <skinnedMesh
        geometry={n.Hair_Base_2?.geometry}
        material={m.Scalp_Transparency}
        skeleton={n.Hair_Base_2?.skeleton}
      />
    </group>
  );
}

useGLTF.preload(AVATAR_GLB);
useGLTF.preload(ANIMATION_GLB);

function LoadingAvatar() {
  return (
    <Html center>
      <div className="flex flex-col items-center gap-3 text-white/70">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-white/20 border-t-white/80" />
        <span className="text-xs font-medium tracking-wider uppercase">Loading avatar</span>
      </div>
    </Html>
  );
}

interface AvatarProps {
  isSpeaking?: boolean;
  isListening?: boolean;
}

const Avatar = forwardRef<AvatarHandle, AvatarProps>(
  ({ isSpeaking = false, isListening = false }, ref) => {
    // A ref, not state: the render loop reads this every frame, and cue changes
    // arrive dozens of times per second. State here would re-render the tree.
    const mouthShapeRef = useRef('X');

    useImperativeHandle(ref, () => ({
      setMouthShape: (shape: string) => {
        mouthShapeRef.current = shape;
      },
      reset: () => {
        mouthShapeRef.current = 'X';
      },
    }));

    return (
      <div className="relative flex h-full w-full flex-col items-center justify-end overflow-hidden bg-gradient-to-br from-[#0a0c10] to-[#001c38]">
        <div className="pointer-events-none absolute inset-0 z-0 bg-black/40" />

        <div className="relative h-full w-full">
          <Canvas camera={{ position: [0, 0, 2.3], fov: 30 }} className="z-10 h-full w-full">
            <Environment preset="sunset" environmentIntensity={0.2} />
            <ambientLight intensity={0.15} color="#cce8ff" />
            <spotLight
              position={[1.2, 1.69, 10]}
              intensity={120}
              color="#ffffff"
              castShadow
              shadow-mapSize-width={2048}
              shadow-mapSize-height={2048}
              shadow-camera-far={15}
              shadow-camera-near={0.3}
            />
            <spotLight
              position={[-20, -1.9, 5]}
              intensity={600}
              color="#f5e4f4"
              angle={Math.PI / 5}
              penumbra={0.7}
            />
            <spotLight
              position={[0, -1.53, 4.09]}
              intensity={20}
              color="#f2d3f1"
              angle={Math.PI / 5.6}
              penumbra={0.8}
            />
            <spotLight
              position={[3, 1, 5]}
              intensity={15}
              color="#f5e4f4"
              angle={Math.PI / 6}
              penumbra={0.5}
            />

            <Suspense fallback={<LoadingAvatar />}>
              <AvatarModel isSpeaking={isSpeaking} mouthShapeRef={mouthShapeRef} />
            </Suspense>

            <OrbitControls
              enableZoom={false}
              enablePan={false}
              target={[0, 1.5, 0]}
              minPolarAngle={Math.PI / 2.2}
              maxPolarAngle={Math.PI / 2.2}
              minAzimuthAngle={-Math.PI / 8}
              maxAzimuthAngle={Math.PI / 8}
            />
          </Canvas>
        </div>

        <div className="pointer-events-none absolute bottom-0 left-0 z-20 h-32 w-full bg-gradient-to-t from-[#001c38] to-transparent" />

        {isListening && (
          <div className="absolute bottom-24 left-1/2 z-30 -translate-x-1/2">
            <div className="flex items-center gap-2 rounded-full border border-white/20 bg-white/10 px-4 py-2 backdrop-blur-xl">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-pulse-ring rounded-full bg-red-400" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-red-500" />
              </span>
              <span className="text-[11px] font-semibold tracking-wider text-white/80 uppercase">
                Listening
              </span>
            </div>
          </div>
        )}
      </div>
    );
  }
);

Avatar.displayName = 'Avatar';

export default Avatar;

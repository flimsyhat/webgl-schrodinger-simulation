// ------
// Runge-Kutta method (RK4) for approximating the evolution of the time-dependent Schrodinger equation
//
// To Do:
// - Make a few potentials
//    [x] Double slit
//    [ ] Reflection / Transmission / Tunneling
//    [ ] Exponential well (harmonic oscillator)
// - Arbitrary initial wave position/momentum
//
// Note: ideally, we create an instance of a simulation by passing it a potential, initial wavefunction, target canvas, time step size, animation tick limit, and shading style (phase vs. real & complex components vs. envelope)
//
// For 3D -- use final wave texture as a displacement map
// ------

const canvas = document.getElementById('glCanvas');

//using regl, a functional wrapper for webgl, to make everything simpler
const regl = createREGL(
  Object.assign({canvas: canvas, 
                 //necessary for using framebuffers
                 extensions: 'OES_texture_float'}));

function makeFrameBuffer() {
  // returns a framebuffer, a stored frame that doesn't get rendered. we will use a buffer to hold the result of each step of the approximation, and only render the final frame
  let fbo_texture = regl.texture({
    shape: [canvas.width, canvas.height, 4], // same size as the canvas, x and y are power of 2 (not sure if that's necessary, but typically textures are a power of 2), z is 4 because we're using rgba color
    type: 'float' // because it will be storing color floats
  })
  
  return regl.framebuffer({
      color: fbo_texture,
      depth: false,
      stencil: false
    });
}

// for each step of the approximation, we create a buffer to hold the results
let initialBuffer   = makeFrameBuffer();
let k1Buffer        = makeFrameBuffer();
let k2Buffer        = makeFrameBuffer();
let k3Buffer        = makeFrameBuffer();
let k4Buffer        = makeFrameBuffer();
let kCombined       = makeFrameBuffer();

// we'll also create a buffer to hold the potential
let potentialBuffer = makeFrameBuffer();

// time step size
const dt = 0.25; // arbitrary, but anything higher tends to blow up
// space step size
const dx = 1.0 / (canvas.width / 2); // 2px step
let time = 0;

const setupDefault = regl({
  // all the uniforms shared by the intermediate wavefunction shaders, like the canvas resolution and defined potential
  uniforms: {
    u_resolution: ctx => [ctx.framebufferWidth,ctx.framebufferHeight],
    dx: dx,
    dt: dt,
    time: time,
    potential_texture: potentialBuffer,
    wave_texture: initialBuffer
  }
});

function switchTexture(initialTexture, t) {
  // after the first frame, switch the simulation to use the output texture as input
  if (t > 1) {
    return kCombined;
  }
  else {
    return initialTexture;
  }
}

// utility function, included in shaders k1, k2, k3, and k4
const divide_by_sqrt_neg_one = `
  vec2 divide_by_sqrt_neg_one(vec2 c) { 
      // divide by sqrt(-1), ie. rotate 270 deg
      return vec2(c.y, -c.x);
    }
`
// function for computing the approximation step, included in shaders k1, k2, k3, and k4
const compute_k =`
    vec2 k(vec2 p) {
      vec2 psi_initial = psi(p);
      vec2 psi_y_inc = psi(p + vec2(0,dx));
      vec2 psi_x_inc = psi(p + vec2(dx,0));
      vec2 psi_y_dec = psi(p - vec2(0,dx));
      vec2 psi_x_dec = psi(p - vec2(dx,0));
      vec2 laplacian = psi_y_inc + psi_x_inc + psi_y_dec + psi_x_dec - (4.0 * psi_initial);
    return divide_by_sqrt_neg_one(-laplacian + potential(p) * psi_initial);
  }
`

const two_slit_potential = regl({
   uniforms: { // inputs to the shader
     u_resolution: ctx => [ctx.framebufferWidth,ctx.framebufferHeight],
     dx: dx
  },
  
  vert: `
    precision mediump float;
    attribute vec2 position;
    void main () {
      gl_Position = vec4(position, 0, 1);
    }`,

  frag: `
    precision mediump float;
    uniform vec2 u_resolution;
    uniform float dx;

    // defined potential (here we have a square potential around the canvas, and a two slit thing in the center)
    float potential(vec2 p) {
      return float(p.y > (1. - 2.*dx) || p.y < (2.*dx) || p.x > (1. - 2.*dx) || p.x < (2.*dx))
                   + 3.0 * float(p.y < (0.5 + dx) && p.y > (0.5 - dx))
                   * float(p.x < 0.4 || p.x > 0.45) * float(p.x < 0.55 || p.x > 0.6);
    }
    void main () {
      // normalize the coordinates to the resolution of the canvas
      vec2 st = gl_FragCoord.xy / u_resolution;
      
      // set the color
      gl_FragColor = vec4(vec3(potential(st)), 1.0);
    }`,

  attributes: {
    // triangle big enough to fill the screen
    position: [
      -4, 0,
      4, 4,
      4, -4
    ]
  },
  // 3 vertices for triangle
  count: 3,
  
  framebuffer: potentialBuffer, // framebuffer we are writing the output to, storing gl_FragColor
});

let initial_wave_position = [0.5, 0.25];

const create_texture = regl({
  uniforms: { // inputs to the shader
    k_combined_texture: kCombined, // final texture, which will get used as input after the first step
    time: regl.prop('tick'),
    wave_position: initial_wave_position,
  },
  
  vert: `
    precision mediump float;
    attribute vec2 position;
    void main () {
      gl_Position = vec4(position, 0, 1);
    }`,

  frag: `
    precision mediump float;
    uniform sampler2D k_combined_texture;
    uniform vec2 u_resolution;
    uniform int time;
    uniform vec2 wave_position;

    // Initial wavefunction, nondispersive packet with some arbitrary values (we can adjust later)

    #define k 100.0 // Frequency
    #define sigma 70.0 // Width of the envelope
    #define length2(p) dot(p,p)

    vec2 initial_wavefunction(vec2 p) {
      // the function returns a vec2 where the first component is real and the second is imaginary
      return exp(-sigma * length2(p - wave_position)) * vec2(cos(k * (p.y)),  sin(k * (p.y)));
    }

    // The approximated wavefunction evolution, which we switch to after the first step
    #define updated_wavefunction(p) texture2D(k_combined_texture, p).xy

    void main () {
      // normalize the coordinates to the resolution of the canvas
      vec2 st = gl_FragCoord.xy / u_resolution;
      
      // if we're past the first evolution step, use evolved wavefunction as input for the next step of the evolution
      vec2 wavefunction = (time > 1) ? updated_wavefunction(st) : initial_wavefunction(st);
      
      // set the color -- we are storing the real component in the R channel and the imaginary component in the G channel
      gl_FragColor = vec4(wavefunction, 0.0, 1.0);
    }`,

  attributes: {
    // triangle big enough to fill the screen
    position: [
      -4, 0,
      4, 4,
      4, -4
    ]
  },
  // 3 vertices for triangle
  count: 3,
  
  framebuffer: initialBuffer, // framebuffer we are writing the output to, storing gl_FragColor
});

const k1 = regl({
vert:`
    precision mediump float;
    attribute vec2 position;
    void main () {
      gl_Position = vec4(position, 0, 1);
    }`,

  frag: `
    precision mediump float;
    uniform sampler2D wave_texture;
    uniform sampler2D potential_texture;
    uniform vec2 u_resolution;
    uniform float dt;
    uniform float dx;

    #define initial_wavefunction(p) texture2D(wave_texture, p).xy
    #define potential(p) texture2D(potential_texture, p).x
    #define psi(p) (initial_wavefunction(p))
    
    ${divide_by_sqrt_neg_one}
    ${compute_k}

    void main () {
      vec2 st = gl_FragCoord.xy / u_resolution;
      gl_FragColor = vec4(k(st), 0.0, 1.0);
    }`,
  
  attributes: {
    position: [
      -4, 0,
      4, 4,
      4, -4
    ]
  },
  count: 3,
  
  framebuffer: k1Buffer,
});

const k2 = regl({
  uniforms: {
    k1_texture: k1Buffer,
  },
  
  vert:`
    precision mediump float;
    attribute vec2 position;
    void main () {
      gl_Position = vec4(position, 0, 1);
    }`,

  frag: `
    precision mediump float;
    uniform sampler2D wave_texture;
    uniform sampler2D potential_texture;
    uniform sampler2D k1_texture;
    uniform vec2 u_resolution;
    uniform float dt;
    uniform float dx;

    #define wavefunction(p) texture2D(wave_texture, p).xy
    #define potential(p) texture2D(potential_texture, p).x
    #define k1(p) texture2D(k1_texture, p).xy
    #define psi(p) (wavefunction(p) + 0.5 * dt * k1(p))
    
    ${divide_by_sqrt_neg_one}
    ${compute_k}

    void main () {
      vec2 st = gl_FragCoord.xy / u_resolution;
      gl_FragColor = vec4(k(st), 0.0, 1.0);
    }`,
  
  attributes: {
    position: [
      -4, 0,
      4, 4,
      4, -4
    ]
  },
  count: 3,
  
  framebuffer: k2Buffer,
});

const k3 = regl({
  uniforms: {
    k2_texture: k2Buffer,
  },
  
  vert:`
    precision mediump float;
    attribute vec2 position;
    void main () {
      gl_Position = vec4(position, 0, 1);
    }`,

  frag: `
    precision mediump float;
    uniform sampler2D wave_texture;
    uniform sampler2D potential_texture;
    uniform sampler2D k2_texture;
    uniform vec2 u_resolution;
    uniform float dt;
    uniform float dx;

    #define wavefunction(p) texture2D(wave_texture, p).xy
    #define potential(p) texture2D(potential_texture, p).x
    #define k2(p) texture2D(k2_texture, p).xy
    #define psi(p) (wavefunction(p) + 0.5 * dt * k2(p))
    
    ${divide_by_sqrt_neg_one}
    ${compute_k}

    void main () {
      vec2 st = gl_FragCoord.xy / u_resolution;
      gl_FragColor = vec4(k(st), 0.0, 1.0);
    }`,

  attributes: {
    position: [
      -4, 0,
      4, 4,
      4, -4
    ]
  },
  count: 3,
  
  framebuffer: k3Buffer,
});

const k4 = regl({
  uniforms: {
    k3_texture: k3Buffer,
  },
  
  vert:`
    precision mediump float;
    attribute vec2 position;
    void main () {
      gl_Position = vec4(position, 0, 1);
    }`,

  frag: `
    precision mediump float;
    uniform sampler2D wave_texture;
    uniform sampler2D potential_texture;
    uniform sampler2D k3_texture;
    uniform vec2 u_resolution;
    uniform float dt;
    uniform float dx;

    #define wavefunction(p) texture2D(wave_texture, p).xy
    #define potential(p) texture2D(potential_texture, p).x
    #define k3(p) texture2D(k3_texture, p).xy
    #define psi(p) (wavefunction(p) + dt * k3(p))
    
    ${divide_by_sqrt_neg_one}
    ${compute_k}

    void main () {
      vec2 st = gl_FragCoord.xy / u_resolution;
      gl_FragColor = vec4(k(st), 0.0, 1.0);
    }`,

  attributes: {
    position: [
      -4, 0,
      4, 4,
      4, -4
    ]
  },
  count: 3,
  
  framebuffer: k4Buffer,
});

const combine_k = regl({
 uniforms: {
    k1_texture: k1Buffer,
    k2_texture: k2Buffer,
    k3_texture: k3Buffer,
    k4_texture: k4Buffer,
  },
  
  vert:`
    precision mediump float;
    attribute vec2 position;
    void main () {
      gl_Position = vec4(position, 0, 1);
    }`,

  frag: `
    precision mediump float;
    uniform sampler2D wave_texture;
    uniform sampler2D k1_texture;
    uniform sampler2D k2_texture;
    uniform sampler2D k3_texture;
    uniform sampler2D k4_texture;
    uniform vec2 u_resolution;
    uniform float dt;

    #define psi(p) texture2D(wave_texture, p).xy
    #define k1(p) texture2D(k1_texture, p).xy
    #define k2(p) texture2D(k2_texture, p).xy
    #define k3(p) texture2D(k3_texture, p).xy
    #define k4(p) texture2D(k4_texture, p).xy

    #define combined_k(p) (psi(p) + dt * (k1(p) + 2.*k2(p) + 2.*k3(p) + k4(p))/6.)

    void main () {
      vec2 st = gl_FragCoord.xy / u_resolution;
      gl_FragColor = vec4(combined_k(st), 0.0, 1.0);
    }`,

  attributes: {
    position: [
      -4, 0,
      4, 4,
      4, -4
    ]
  },
  count: 3,
  
  framebuffer: kCombined,
});

const draw_frame = regl({
  // No framebuffer, because we are rendering a frame
  uniforms: {
    u_resolution: ctx => [ctx.framebufferWidth,ctx.framebufferHeight],
    k_combined_texture: kCombined,
    potential_texture: potentialBuffer,
  },
  
  vert:`
    precision mediump float;
    attribute vec2 position;
    void main () {
      gl_Position = vec4(position, 0, 1);
    }`,

  frag: `
    precision mediump float;
    uniform sampler2D k_combined_texture;
    uniform sampler2D potential_texture;
    uniform vec2 u_resolution;

    #define k_combined(p) texture2D(k_combined_texture, p).xy
    #define potential(p) texture2D(potential_texture, p).x
    #define PI 3.141592653589793
    #define hue2rgb(h) clamp(abs(mod(6.*(h)+vec3(0,4,2),6.)-3.)-1.,0.,1.)

    void main () {
      vec2 st = gl_FragCoord.xy / u_resolution;
      vec2 v = k_combined(st);
      // gl_FragColor = vec4(1.5 * length(v) * hue2rgb(atan(v.y,v.x)/(2.*PI)) + 0.25*potential(st), 1.0) + vec4(potential(st), potential(st), potential(st), 1.0);
      gl_FragColor = vec4(0.0, v, 1.0)
                     + vec4(vec3(potential(st)), 1.0);
    }`,

  attributes: {
    position: [
      -4, 0,
      4, 4,
      4, -4
    ]
  },
  count: 3,
});

const animationTickLimit = 2000; // -1 disables

// Stuff for displaying frame number

document.getElementById("frame_limit").innerHTML = animationTickLimit;
const frame_display = document.getElementById("frame");

// Create potential texture
two_slit_potential();

// main animation loop
const frameLoop = regl.frame(({ tick }) => {
    frame_display.innerHTML = tick;
	// clear the buffer
	regl.clear({
		// background color (black)
		color: [0, 0, 0, 1],
		depth: 1,
	});
    let time = tick;
    // step through approximation, rendering each step to a frame buffer 
    setupDefault({}, () => {
      create_texture({tick: tick});
      k1();
      k2();
      k3();
      k4();
      combine_k();
    })
    // draw a frame to the screen from the frame buffer
    draw_frame();

	// simple way of stopping the animation after a few ticks
	if (tick === animationTickLimit) {
		console.log(`Hit tick ${tick}, canceling animation loop`);
		// cancel this loop
		frameLoop.cancel();
	}
});
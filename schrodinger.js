const canvas = document.getElementById('glCanvas')

const regl = createREGL(
  Object.assign({canvas: canvas, extensions: 'OES_texture_float'}));

function makeFrameBuffer() {
  let fbo_texture = regl.texture({
    shape: [512, 512, 4],
    type: 'float'
  })
  
  return regl.framebuffer({
      color: fbo_texture,
      depth: false,
      stencil: false
    });
}

let initialBuffer = makeFrameBuffer();
let k1Buffer      = makeFrameBuffer();
let k2Buffer      = makeFrameBuffer();
let k3Buffer      = makeFrameBuffer();
let k4Buffer      = makeFrameBuffer();
let kCombined     = makeFrameBuffer();

// Step size
const dt = 0.05;

const create_texture = regl({
  framebuffer: initialBuffer,
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

    // Initial wavefunction, nondispersive packet with some arbitrary values (we can adjust later)

    #define k 100.0 // Frequency
    #define sigma 50.0 // Width of the envelope
    #define length2(p) dot(p,p)

    vec2 initial_wavefunction(vec2 p) {
      return exp(-sigma * length2(p - vec2(0.5, 0.5))) * vec2(cos(k * (p.x)),  sin(k * (p.x)));
      // the function returns a vec2 where the first component is real and the second is imaginary
    }

    // The approximated evolution, which we switch to after the first step
    #define updated_wavefunction(p) texture2D(k_combined_texture, p).xy

    void main () {
      // normalize the coordinates to the resolution of the canvas
      vec2 st = gl_FragCoord.xy / u_resolution;

      // set the color -- we are storing the real component in the R channel and the complex component in the G channel
      vec2 wave = (time > 1) ? updated_wavefunction(st) : initial_wavefunction(st);
      gl_FragColor = vec4(wave, 0.0, 1.0);
    }`,

  attributes: {
      // triangle big enough to fill the screen
  position: [
    -4, 0,
    4, 4,
    4, -4
  ]
  },

  // 
  uniforms: {
    u_resolution: ctx => [ctx.framebufferWidth,ctx.framebufferHeight],
    k_combined_texture: kCombined,
    time: regl.prop('tick')   
  },

  // 3 vertices for triangle
  count: 3,
});

const k1 = regl({
  framebuffer: k1Buffer,
  vert:`
    precision mediump float;
    attribute vec2 position;
    void main () {
      gl_Position = vec4(position, 0, 1);
    }`,

  frag: `
    precision mediump float;
    uniform sampler2D texture;
    uniform sampler2D k_combined_texture;
    uniform vec2 u_resolution;
    uniform float dt;

    #define initial_wavefunction(p) texture2D(texture, p).xy
    #define psi(p) (initial_wavefunction(p))

    vec2 divide_by_sqrt_neg_one(vec2 c) { /* divide by sqrt(-1), ie. rotate 270 deg */
      return vec2(c.y, -c.x);
    }

    // define square well potential
    float potential(vec2 p) {
      return float(p.y > 0.99 || p.y < 0.01 || p.x > 0.99 || p.x < 0.01);
    }

    vec2 k1(vec2 p) {
      vec2 psi_initial = psi(p);
      vec2 psi_y_inc = psi(p + vec2(0,1));
      vec2 psi_x_inc = psi(p + vec2(1,0));
      vec2 psi_y_dec = psi(p - vec2(0,1));
      vec2 psi_x_dec = psi(p - vec2(1,0));
      vec2 laplacian = psi_y_inc + psi_x_inc + psi_y_dec + psi_x_dec - (4.0 * psi_initial);
      return divide_by_sqrt_neg_one(-laplacian + potential(p) * psi_initial);
    }

    void main () {
      vec2 st = gl_FragCoord.xy / u_resolution;
      gl_FragColor = vec4(k1(st), 0.0, 1.0);
    }`,

  attributes: {
    position: [
      -4, 0,
      4, 4,
      4, -4
    ]
  },
  
  uniforms: {
    u_resolution: ctx => [ctx.framebufferWidth,ctx.framebufferHeight],
    texture: initialBuffer,
    dt : dt 
  },
  
  count: 3,
});

const k2 = regl({
  framebuffer: k2Buffer,
  vert:`
    precision mediump float;
    attribute vec2 position;
    void main () {
      gl_Position = vec4(position, 0, 1);
    }`,

  frag: `
    precision mediump float;
    uniform sampler2D texture;
    uniform sampler2D k1_texture;
    uniform vec2 u_resolution;
    uniform float dt;

    #define wavefunction(p) texture2D(texture, p).xy
    #define k1(p) texture2D(k1_texture, p).xy
    #define psi(p) (wavefunction(p) + 0.5 * dt * k1(p))

    vec2 divide_by_sqrt_neg_one(vec2 c) { /* divide by sqrt(-1), ie. rotate 270 deg */
      return vec2(c.y, -c.x);
    }

    float potential(vec2 p) {
      return float(p.y > 0.99 || p.y < 0.01 || p.x > 0.99 || p.x < 0.01);
    }

    vec2 k2(vec2 p) {
      vec2 psi_initial = psi(p);
      vec2 psi_y_inc = psi(p + vec2(0,1));
      vec2 psi_x_inc = psi(p + vec2(1,0));
      vec2 psi_y_dec = psi(p - vec2(0,1));
      vec2 psi_x_dec = psi(p - vec2(1,0));
      vec2 laplacian = psi_y_inc + psi_x_inc + psi_y_dec + psi_x_dec - (4.0 * psi_initial);
      return divide_by_sqrt_neg_one(-laplacian + potential(p) * psi_initial);
    }

    void main () {
      vec2 st = gl_FragCoord.xy / u_resolution;
      gl_FragColor = vec4(k2(st), 0.0, 1.0);
    }`,

  attributes: {
    position: [
      -4, 0,
      4, 4,
      4, -4
    ]
  },
  
  uniforms: {
    u_resolution: ctx => [ctx.framebufferWidth,ctx.framebufferHeight],
    k1_texture: k1Buffer,
    texture: initialBuffer,
    dt : dt
  },
  
  count: 3,
});

const k3 = regl({
  framebuffer: k3Buffer,
  vert:`
    precision mediump float;
    attribute vec2 position;
    void main () {
      gl_Position = vec4(position, 0, 1);
    }`,

  frag: `
    precision mediump float;
    uniform sampler2D texture;
    uniform sampler2D k2_texture;
    uniform vec2 u_resolution;
    uniform float dt;

    #define wavefunction(p) texture2D(texture, p).xy
    #define k2(p) texture2D(k2_texture, p).xy
    #define psi(p) (wavefunction(p) + 0.5 * dt * k2(p))

    vec2 divide_by_sqrt_neg_one(vec2 c) { /* divide by sqrt(-1), ie. rotate 270 deg */
      return vec2(c.y, -c.x);
    }

    float potential(vec2 p) {
      return float(p.y > 0.99 || p.y < 0.01 || p.x > 0.99 || p.x < 0.01);
    }

    vec2 k3(vec2 p) {
      vec2 psi_initial = psi(p);
      vec2 psi_y_inc = psi(p + vec2(0,1));
      vec2 psi_x_inc = psi(p + vec2(1,0));
      vec2 psi_y_dec = psi(p - vec2(0,1));
      vec2 psi_x_dec = psi(p - vec2(1,0));
      vec2 laplacian = psi_y_inc + psi_x_inc + psi_y_dec + psi_x_dec - (4.0 * psi_initial);
      return divide_by_sqrt_neg_one(-laplacian + potential(p) * psi_initial);
    }

    void main () {
      vec2 st = gl_FragCoord.xy / u_resolution;
      gl_FragColor = vec4(k3(st), 0.0, 1.0);
    }`,

  attributes: {
    position: [
      -4, 0,
      4, 4,
      4, -4
    ]
  },
  
  uniforms: {
    u_resolution: ctx => [ctx.framebufferWidth,ctx.framebufferHeight],
    k2_texture: k2Buffer,
    texture: initialBuffer,
    dt : dt
  },
  
  count: 3,
});

const k4 = regl({
  framebuffer: k4Buffer,
  vert:`
    precision mediump float;
    attribute vec2 position;
    void main () {
      gl_Position = vec4(position, 0, 1);
    }`,

  frag: `
    precision mediump float;
    uniform sampler2D texture;
    uniform sampler2D k3_texture;
    uniform vec2 u_resolution;
    uniform float dt;

    #define wavefunction(p) texture2D(texture, p).xy
    #define k3(p) texture2D(k3_texture, p).xy
    #define psi(p) (wavefunction(p) + dt * k3(p))

    vec2 divide_by_sqrt_neg_one(vec2 c) { /* divide by sqrt(-1), ie. rotate 270 deg */
      return vec2(c.y, -c.x);
    }

    float potential(vec2 p) {
      return float(p.y > 0.99 || p.y < 0.01 || p.x > 0.99 || p.x < 0.01);
    }

    vec2 k4(vec2 p) {
      vec2 psi_initial = psi(p);
      vec2 psi_y_inc = psi(p + vec2(0,1));
      vec2 psi_x_inc = psi(p + vec2(1,0));
      vec2 psi_y_dec = psi(p - vec2(0,1));
      vec2 psi_x_dec = psi(p - vec2(1,0));
      vec2 laplacian = psi_y_inc + psi_x_inc + psi_y_dec + psi_x_dec - (4.0 * psi_initial);
      return divide_by_sqrt_neg_one(-laplacian + potential(p) * psi_initial);
    }

    void main () {
      vec2 st = gl_FragCoord.xy / u_resolution;
      gl_FragColor = vec4(k4(st), 0.0, 1.0);
    }`,

  attributes: {
    position: [
      -4, 0,
      4, 4,
      4, -4
    ]
  },
  
  uniforms: {
    u_resolution: ctx => [ctx.framebufferWidth,ctx.framebufferHeight],
    k3_texture: k3Buffer,
    texture: initialBuffer,
    dt : dt
  },
  
  count: 3,
});

const combine_k = regl({
  framebuffer: kCombined,
  vert:`
    precision mediump float;
    attribute vec2 position;
    void main () {
      gl_Position = vec4(position, 0, 1);
    }`,

  frag: `
    precision mediump float;
    uniform sampler2D texture;
    uniform sampler2D k1_texture;
    uniform sampler2D k2_texture;
    uniform sampler2D k3_texture;
    uniform sampler2D k4_texture;
    uniform vec2 u_resolution;
    uniform float dt;

    #define psi(p) texture2D(texture, p).xy
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
  
  uniforms: {
    u_resolution: ctx => [ctx.framebufferWidth,ctx.framebufferHeight],
    k1_texture: k1Buffer,
    k2_texture: k2Buffer,
    k3_texture: k3Buffer,
    k4_texture: k4Buffer,
    texture: initialBuffer,
    dt : dt
  },
  
  count: 3,
});

const draw_frame = regl({
  vert:`
    precision mediump float;
    attribute vec2 position;
    void main () {
      gl_Position = vec4(position, 0, 1);
    }`,

  frag: `
    precision mediump float;
    uniform sampler2D k_combined_texture;
    uniform vec2 u_resolution;

    #define k_combined(p) texture2D(k_combined_texture, p).xy
    #define PI 3.141592653589793
    #define hue2rgb(h) clamp(abs(mod(6.*(h)+vec3(0,4,2),6.)-3.)-1.,0.,1.)

    float potential(vec2 p) {
      return float(p.y > 0.99 || p.y < 0.01 || p.x > 0.99 || p.x < 0.01);
    }

    void main () {
      vec2 st = gl_FragCoord.xy / u_resolution;
      vec2 v = k_combined(st);
      gl_FragColor = vec4(1.5 * length(v) * hue2rgb(atan(v.y,v.x)/(2.*PI)), 1.0)
                     + vec4(potential(st), potential(st), potential(st), 1.0);
    }`,

  attributes: {
    position: [
      -4, 0,
      4, 4,
      4, -4
    ]
  },
  
  uniforms: {
    u_resolution: ctx => [ctx.framebufferWidth,ctx.framebufferHeight],
    k_combined_texture: kCombined
  },
  
  count: 3,
});

const animationTickLimit = 1000; // -1 disables
if (animationTickLimit >= 0) {
  console.log(`Limiting to ${animationTickLimit} ticks`);
}

const frameLoop = regl.frame(({ tick }) => {
    console.log(tick)
	// clear the buffer
	regl.clear({
		// background color (black)
		color: [0, 0, 0, 1],
		depth: 1,
	});
    create_texture({tick: tick});
    k1();
    k2();
    k3();
    k4();
    combine_k();
    draw_frame();

	// simple way of stopping the animation after a few ticks
	if (tick === animationTickLimit) {
		console.log(`Hit tick ${tick}, canceling animation loop`);
		// cancel this loop
		frameLoop.cancel();
	}
});

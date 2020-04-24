// ------
// Euler method (RK4) for approximating the evolution of the time-dependent Schrodinger equation
// ------

//using regl, a functional wrapper for webgl, to make everything simpler
const regl = createREGL(
  Object.assign({canvas: document.getElementById('glCanvas'), 
                 //necessary for using framebuffers
                 extensions: 'OES_texture_float'}));

function makeFrameBuffer() {
  // returns a framebuffer, a stored frame that doesn't get rendered. we will use a buffer to hold the result of each step of the approximation, and only render the final frame
  let fbo_texture = regl.texture({
    shape: [512, 512, 4], // same size as the canvas, x and y are power of 2 (not sure if that's necessary, but typically textures are a power of 2), z is 4 because we're using rgba color
    type: 'float' // because it will be storing color floats
  })
  
  return regl.framebuffer({
      color: fbo_texture,
      depth: false,
      stencil: false
    });
}

// for each step of the approximation, we create a buffer to hold the results
let initialBuffer = makeFrameBuffer();
let finalBuffer   = makeFrameBuffer();

// time step size
const dt = 0.05;

const create_texture = regl({
  uniforms: { // inputs to the shader
    u_resolution: ctx => [ctx.framebufferWidth,ctx.framebufferHeight],
    final_texture: finalBuffer, // final texture, which will get used as input after the first step
    time: regl.prop('tick')   
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
    uniform sampler2D final_texture;
    uniform int time;

    // Initial wavefunction, nondispersive packet with some arbitrary values (we can adjust later)

    #define k 100.0 // Frequency
    #define sigma 50.0 // Width of the envelope
    #define length2(p) dot(p,p)

    vec2 initial_wavefunction(vec2 p) {
      // the function returns a vec2 where the first component is real and the second is "complex"
      return exp(-sigma * length2(p - vec2(0.5, 0.5))) * vec2(cos(k * (p.x)),  sin(k * (p.x)));
    }

    // The approximated wavefunction evolution, which we switch to after the first step
    #define updated_wavefunction(p) texture2D(final_texture, p).xy

    void main () {
      // normalize the coordinates to the resolution of the canvas
      vec2 st = gl_FragCoord.xy / u_resolution;
      
      // if we're past the first evolution step, use evolved wavefunction as input for the next step of the evolution
      vec2 wavefunction = (time > 1) ? updated_wavefunction(st) : initial_wavefunction(st);
      
      // set the color -- we are storing the real component in the R channel and the complex component in the G channel
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
  uniforms: {
    u_resolution: ctx => [ctx.framebufferWidth,ctx.framebufferHeight],
    texture: initialBuffer,
    dt : dt 
  },
  
  vert:`
    precision mediump float;
    attribute vec2 position;
    void main () {
      gl_Position = vec4(position, 0, 1);
    }`,

  frag: `
    precision mediump float;
    uniform sampler2D texture;
    uniform vec2 u_resolution;
    uniform float dt;

    #define psi(p) texture2D(texture, p).xy

    vec2 divide_by_sqrt_neg_one(vec2 c) { 
      // divide by sqrt(-1), ie. rotate 270 deg
      return vec2(c.y, -c.x);
    }

    // define square well potential, returns 1.0 at the edges and 0.0 everwhere else
    float potential(vec2 p) {
      return float(p.y > 0.99 || p.y < 0.01 || p.x > 0.99 || p.x < 0.01);
    }

    const vec2 mixing = vec2(-1.0, +1.0);

    vec2 wavefunction(vec2 p) {
      float real = psi(p).x
                   - (0.5 * ( psi(p + vec2(0,1)).y
                            + psi(p + vec2(1,0)).y
                            + psi(p - vec2(0,1)).y
                            + psi(p - vec2(1,0)).y
                             - 4.0 * psi(p).y )
                   + potential(p) * psi(p).y) * dt;
      
      float imaginary = psi(p).y
                   - (0.5 * ( psi(p + vec2(0,1)).x
                            + psi(p + vec2(1,0)).x
                            + psi(p - vec2(0,1)).x
                            + psi(p - vec2(1,0)).x
                             - 4.0 * psi(p).x )
                   + potential(p) * psi(p).x) * dt;

      return vec2(real, imaginary);
    }

    void main () {
      vec2 st = gl_FragCoord.xy / u_resolution;
      gl_FragColor = vec4(wavefunction(st), 0.0, 1.0);
    }`,
  
  attributes: {
    position: [
      -4, 0,
      4, 4,
      4, -4
    ]
  },
  count: 3,
  
  framebuffer: finalBuffer,
});

const draw_frame = regl({
  // No framebuffer, because we are rendering a frame
  uniforms: {
    u_resolution: ctx => [ctx.framebufferWidth,ctx.framebufferHeight],
    final_texture: finalBuffer
  },
  
  vert:`
    precision mediump float;
    attribute vec2 position;
    void main () {
      gl_Position = vec4(position, 0, 1);
    }`,

  frag: `
    precision mediump float;
    uniform sampler2D final_texture;
    uniform vec2 u_resolution;

    #define wavefunction(p) texture2D(final_texture, p).xy

    float potential(vec2 p) {
      return float(p.y > 0.99 || p.y < 0.01 || p.x > 0.99 || p.x < 0.01);
    }

    void main () {
      vec2 st = gl_FragCoord.xy / u_resolution;
      gl_FragColor = vec4(wavefunction(st), 0.0, 1.0)
                     + vec4(potential(st), potential(st), potential(st), 1.0);
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

const animationTickLimit = 1; // -1 disables

if (animationTickLimit >= 0) {
  console.log(`Limiting to ${animationTickLimit} ticks`);
}

// main animation loop
const frameLoop = regl.frame(({ tick }) => {
    console.log(tick)
	// clear the buffer
	regl.clear({
		// background color (black)
		color: [0, 0, 0, 1],
		depth: 1,
	});
    create_texture({tick: tick});
    draw_frame();

	// simple way of stopping the animation after a few ticks
	if (tick === animationTickLimit) {
		console.log(`Hit tick ${tick}, canceling animation loop`);
		// cancel this loop
		frameLoop.cancel();
	}
});
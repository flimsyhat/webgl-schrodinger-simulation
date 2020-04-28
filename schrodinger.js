// ------
// Runge-Kutta method (RK4) for approximating the evolution of the time-dependent Schrodinger equation
//
// To Do:
// - Make a few potentials (and ability to select between them)
//    [x] Double slit
//    [ ] Reflection / Transmission / Tunneling
//    [ ] Exponential well (harmonic oscillator)
// - Fix initial wave behavior -- right now if the wave is created and part of it intersects the potential, we get really fast waves. Need to set those inersections to 0 initially
// - Buttons to pause and restart
//
// Note: ideally, we create an instance of a simulation by passing it a potential, initial wavefunction, target canvas, time step size, animation tick limit, and shading style (phase vs. real & complex components vs. envelope)
//
// For 3D -- maybe use final wave texture as a displacement map?
//
// Longer-term: Ability to draw custom potentials ?
// ------


const glCanvas  = document.getElementById('glCanvas');
const topCanvas = document.getElementById('2dCanvas');

// ------
// Initial wave position, angle, and some conversion functions used by the mouse events when updating them
// ------

// default wave direction is vertical, will be modified by mouseclick events
var wave_angle = Math.PI/2;
var wave_angle_components = angle_components(angle);

// initial wave position, will be modified mouseclick events
var wave_position = [0.5, 0.25];

// time step size -- arbitrary, but anything higher tends to blow up
const dt = 0.25;
// space step size (2px step)
const dx = 1.0 / (glCanvas.width / 2);

// used to keep track of time since last mouseclick, since the animation loop continues incrementing regardless
var elapsedTime = 0;
// ------

//using regl, a functional wrapper for webgl, to make everything simpler
const regl = createREGL(
  Object.assign({canvas: glCanvas, 
                 //necessary for using framebuffers
                 extensions: 'OES_texture_float'}));

function makeFrameBuffer() {
  // returns a framebuffer, a stored frame that doesn't get rendered. we will use a buffer to hold the result of each step of the approximation, and only render the final frame
  let fbo_texture = regl.texture({
    shape: [glCanvas.width, glCanvas.height, 4], // same size as the canvas, x and y are power of 2 (not sure if that's necessary, but typically textures are a power of 2), z is 4 because we're using rgba color
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

const setupDefault = regl({
  // all the uniforms shared by the intermediate wavefunction shaders, like the canvas resolution and defined potential
  uniforms: {
    u_resolution: ctx => [ctx.framebufferWidth,ctx.framebufferHeight],
    dx: dx,
    dt: dt,
    time: elapsedTime,
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

const create_texture = regl({
  uniforms: {
    k_combined_texture: kCombined, // final texture, which will get used as input after the first step
    time: regl.prop('time'),
    wave_position: regl.prop('wave_position'),
    wave_angles: regl.prop('wave_angles')
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
    uniform vec2 wave_angles;

    // Initial wavefunction, nondispersive packet with some arbitrary values (we can adjust later)

    #define k 100.0 // Frequency
    #define sigma 100.0 // Width of the envelope
    #define length2(p) dot(p,p)

    vec2 initial_wavefunction(vec2 p) {
      // the function returns a vec2 where the first component is real and the second is imaginary
      return exp(-sigma * length2(p - wave_position)) * vec2(cos(k * (wave_angles.y * p.y + wave_angles.x * p.x)),  sin(k * (wave_angles.y * p.y + wave_angles.x * p.x)));
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
    position: [
      -4, 0,
      4, 4,
      4, -4
    ]
  },
  count: 3,
  
  framebuffer: initialBuffer,
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

// ------
// MAIN ANIMATION LOOP
// ------

const animationTimeLimit = 2000; // -1 disables

// Stuff for displaying frame number

document.getElementById("frame_limit").innerHTML = animationTimeLimit;
const frame_display = document.getElementById("frame");

// Create potential texture
two_slit_potential();

// main animation loop
function frameLoop() {
  let frameLoop = regl.frame(() => {
    
    if (MOUSE_DOWN) {
      elapsedTime = 0;
    }
        
    elapsedTime += 1;
    frame_display.innerHTML = elapsedTime;
	
    // clear the buffer
	regl.clear({
		// background color (black)
		color: [0, 0, 0, 1],
		depth: 1,
	});
    
    // step through approximation, rendering each step to a frame buffer 
    setupDefault({}, () => {
      create_texture({time: elapsedTime, wave_position: wave_position, wave_angles: wave_angle_components});
      k1();
      k2();
      k3();
      k4();
      combine_k();
    })
    
    // draw a frame to the screen from the frame buffer
    draw_frame();

	// simple way of looping the animation after a certain time
	if (elapsedTime === animationTimeLimit) {
      elapsedTime = 0;
	}
})};

frameLoop();

// ------
// MOUSE EVENTS
//
// This is still messy, particularly the logic for clearing the 2d canvas
// ------

let MOUSE_DOWN = false;

function getCursorPosition(canvas, event) {
  const rect = canvas.getBoundingClientRect()
  let x = (event.clientX - rect.left) / glCanvas.width;
  let y = 1.0 - (event.clientY - rect.top) / glCanvas.height;
  return [x, y]
}

topCanvas.addEventListener('mousedown', function(e) {
  MOUSE_DOWN = true;
  wave_position = getCursorPosition(topCanvas, e);
})

topCanvas.addEventListener('mouseup', function(e) {
  MOUSE_DOWN = false;
  clear_2d_canvas(topCanvas)
})

topCanvas.addEventListener ("mouseout", function(e) {
  // if mouse is down when the mouse leaves the canvas, set the MOUSE_DOWN flag to false and start the simulation
  if (!MOUSE_DOWN) {
    return;
  }
  MOUSE_DOWN = false;
  clear_2d_canvas(topCanvas)
})


window.addEventListener('mousemove', function(e) { 
  if (!MOUSE_DOWN) {
    return;
  }
  let cursor_position = getCursorPosition(glCanvas, e);
  let min_distance = 0.025;
  if (distance(wave_position, cursor_position) > min_distance) {
    wave_angle = angle(wave_position, cursor_position);
    wave_angle_components = angle_components(angle);
    draw_line(topCanvas, wave_position, cursor_position)
  }
})

// ------
// Conversion functions used by the mouseclick events
// ------

function distance(p_a, p_b) {
  return Math.sqrt((p_a[0] - p_b[0]) ** 2 + (p_a[1] - p_b[1]) ** 2)
}

function angle(p_a, p_b) {
  var dy = p_b[1] - p_a[1];
  var dx = p_b[0] - p_a[0];
  var theta = Math.atan2(dy, dx); // range (-PI, PI]
  return theta;
}

function angle_components(angle) {
  // takes an angle and returns the x and y position on the unit circle
  return [Math.cos(wave_angle), Math.sin(wave_angle)]
}

// ------
// Drawing the dashed line on the 2D top canvas
// ------

function convert_to_canvas_coordinates(coord) {
  // convert from GL coordinates, which range between 0 and 1 and start from bottom left corner, to canvas coordinates, which are in px and start from top left corner
  let x = coord[0] * topCanvas.width;
  let y = (1 - coord[1]) * topCanvas.height;
  return [x, y]
}

function draw_line(canvas, p_a, p_b) {
  p_a = convert_to_canvas_coordinates(p_a)
  p_b = convert_to_canvas_coordinates(p_b)
  let ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, topCanvas.width, topCanvas.height); // erase previous line
  ctx.setLineDash([5, 3]);/*dashes are 5px and spaces are 3px*/
  ctx.beginPath();
  ctx.moveTo(p_a[0],p_a[1]);
  ctx.lineTo(p_b[0],p_b[1]);
  ctx.strokeStyle = 'rgb(255, 255, 255)';
  ctx.stroke();
}

function clear_2d_canvas(canvas) {
  let ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, topCanvas.width, topCanvas.height);
}
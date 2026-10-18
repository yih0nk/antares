// Logs commanded angle and measured potentiometer reading for up to two
// SG92R servos, so antares can calibrate each one as a one-joint robot.
//
// Wiring (see bench/README.md):
//   servo A: signal -> D9,  pot wiper -> A0
//   servo B: signal -> D10, pot wiper -> A1
//   both servos: brown -> GND, red -> 5V (external 5V supply for two)
//
// Protocol: the sketch waits for a 'g' on serial, plays the same
// 60 s command program, prints one CSV line per 20 ms tick, then "#done".
// Send 'g' again for another run.

#include <Servo.h>

const int NUM_SERVOS = 2;            // set to 1 if only servo A is wired
const int SIGNAL_PIN[2] = {9, 10};
const int FEEDBACK_PIN[2] = {A0, A1};
const unsigned long TICK_MS = 20;    // 50 Hz, the servo's own PWM rate
const float LO = 30.0, HI = 150.0;   // stay off the end stops

Servo servos[2];

// Fixed step targets so every run (and every servo) sees the same program.
const float STEPS[] = {90, 130, 50, 110, 70, 140, 40, 100, 60, 120,
                       80, 135, 45, 95, 125, 55, 105, 65, 115, 90};
const int NUM_STEPS = sizeof(STEPS) / sizeof(STEPS[0]);

float command_at(float t) {
  // 0-10 s: calibration holds, 2 s each (maps pot counts to degrees)
  if (t < 10.0) {
    const float holds[5] = {30, 60, 90, 120, 150};
    return holds[(int)(t / 2.0)];
  }
  // 10-26 s: steps every 0.8 s (shows stiffness, damping, latency)
  if (t < 26.0) {
    int i = (int)((t - 10.0) / 0.8);
    return STEPS[i % NUM_STEPS];
  }
  // 26-41 s: 0.5 Hz sine, +-40 deg
  if (t < 41.0) {
    return 90.0 + 40.0 * sin(2.0 * PI * 0.5 * (t - 26.0));
  }
  // 41-56 s: chirp 0.2 -> 2 Hz, +-30 deg (shows lag growing with speed)
  if (t < 56.0) {
    float tau = t - 41.0, f0 = 0.2, f1 = 2.0, dur = 15.0;
    float phase = 2.0 * PI * (f0 * tau + 0.5 * (f1 - f0) / dur * tau * tau);
    return 90.0 + 30.0 * sin(phase);
  }
  // 56-60 s: park
  return 90.0;
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].attach(SIGNAL_PIN[i]);
    servos[i].write(90);
  }
  Serial.println("#ready: send g to start a 60 s run");
}

void run_once() {
  Serial.print("t_ms,cmd_deg");
  for (int i = 0; i < NUM_SERVOS; i++) {
    Serial.print(",raw_");
    Serial.print((char)('a' + i));
  }
  Serial.println();

  unsigned long start = millis();
  unsigned long next = start;
  while (true) {
    unsigned long now = millis();
    if ((long)(now - next) < 0) continue;
    next += TICK_MS;
    float t = (now - start) / 1000.0;
    if (t >= 60.0) break;

    // The servo library takes whole degrees; log exactly what was sent.
    int cmd = (int)(constrain(command_at(t), LO, HI) + 0.5);
    // Read before writing so each line pairs a command with the
    // position the servo had reached when that command was sent.
    int raw[2];
    for (int i = 0; i < NUM_SERVOS; i++) raw[i] = analogRead(FEEDBACK_PIN[i]);
    for (int i = 0; i < NUM_SERVOS; i++) servos[i].write(cmd);

    Serial.print(now - start);
    Serial.print(',');
    Serial.print(cmd);
    for (int i = 0; i < NUM_SERVOS; i++) {
      Serial.print(',');
      Serial.print(raw[i]);
    }
    Serial.println();
  }
  for (int i = 0; i < NUM_SERVOS; i++) servos[i].write(90);
  Serial.println("#done");
}

void loop() {
  if (Serial.available() && Serial.read() == 'g') run_once();
}

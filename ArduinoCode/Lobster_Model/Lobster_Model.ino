#include <Servo.h> 
/*
   Program for Lobster Model for SfN 2022 Presentation.
   The robot does not communicate with the TDT apparatus,
   and their IO port is now used for the lick sensor
   since the current board incorporated the lick controller.
*/
const int PIN_LICK = 9;
const int PIN_ATTACK_IN_MANUAL = 8;

const int PIN_SERVO1 = 12; // Left 
const int PIN_SERVO2 = 11; // Right

const int PIN_SERVO_ON = 13;

Servo myservo1;
Servo myservo2;

// servo angles
// after the motor replacement, the angle value changed.
// these are the old values
//const int servo1_rest = 65;
//const int servo1_attk = 100;

const int servo1_rest = 50;
const int servo1_attk = 85;
const int servo2_rest = 140;
const int servo2_attk = 105;

void setup() 
{
  pinMode(PIN_SERVO_ON, OUTPUT);
  pinMode(PIN_LICK, INPUT);
  pinMode(PIN_ATTACK_IN_MANUAL, INPUT);

  // initialize motor
  myservo1.attach(PIN_SERVO1);
  myservo2.attach(PIN_SERVO2);

  digitalWrite(PIN_SERVO_ON, HIGH);
  delay(50);
  myservo1.write(servo1_rest);
  myservo2.write(servo2_rest);
  delay(200);
  digitalWrite(PIN_SERVO_ON, LOW);

  Serial.begin(9600);
} 
 
 unsigned long firstLickTime;
 bool isArmed = false;
void loop() 
{
  if(!digitalRead(PIN_LICK) && !isArmed)
  {
    isArmed = true;
    firstLickTime = millis();
    Serial.println('a');
  }

  if( (isArmed && (millis()-firstLickTime > 3000)) || digitalRead(PIN_ATTACK_IN_MANUAL) == HIGH) // Attack
  {

    // Perform Attack
    attack();
    
    // Re-organize the claw once again
    myservo1.write(servo1_rest);
    myservo2.write(servo2_rest);

    // Reinitialize
    isArmed = false;
  }
}

void attack()
{
  digitalWrite(PIN_SERVO_ON, HIGH);
  delay(50);
  myservo1.write(servo1_attk);               
  myservo2.write(servo2_attk);
  delay(200);
  myservo1.write(servo1_rest);  
  myservo2.write(servo2_rest);
  delay(200);
  myservo1.write(servo1_attk);               
  myservo2.write(servo2_attk);
  delay(200);
  myservo1.write(servo1_rest);
  myservo2.write(servo2_rest);
  delay(200);
  digitalWrite(PIN_SERVO_ON, LOW);
}

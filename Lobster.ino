#include <Servo.h> 

const int PIN_ATTACK_IN_MANUAL = 8; 
const int PIN_ATTACK_IN = 9;
const int PIN_ATTACK_OUT = 10;

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
  pinMode(PIN_ATTACK_IN, INPUT);
  pinMode(PIN_ATTACK_OUT, OUTPUT);
  pinMode(PIN_SERVO_ON, OUTPUT);

  digitalWrite(PIN_ATTACK_OUT,LOW);

  // initialize motor
  myservo1.attach(PIN_SERVO1);
  myservo2.attach(PIN_SERVO2);

  digitalWrite(PIN_SERVO_ON, HIGH);
  delay(50);
  myservo1.write(servo1_rest);
  myservo2.write(servo2_rest);
  delay(200);
  digitalWrite(PIN_SERVO_ON, LOW);
} 
 
void loop() 
{
  if(digitalRead(PIN_ATTACK_IN) == HIGH || digitalRead(PIN_ATTACK_IN_MANUAL) == HIGH) // Attack
  {
    // Send Attack Info to TDT
    digitalWrite(PIN_ATTACK_OUT,HIGH);

    // Perform Attack
    attack();

    // Send Attack Info
    digitalWrite(PIN_ATTACK_OUT,LOW);
    
    // Re-organize the claw once again
    myservo1.write(servo1_rest);
    myservo2.write(servo2_rest);
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

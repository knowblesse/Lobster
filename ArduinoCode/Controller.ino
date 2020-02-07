/* 
  Created by Knowblesse, 2018-11-27
  Arduino Script for the Lobster Controller
*/

const String START_MSG =\
"Lobsterbot Controller\n \
Version 2.2\n \
Knowblesse 2019";

// Assign Pin Numbers
const int PIN_BLOCK_INPUT = 2; // block toggle switch input
const int PIN_BLOCK_OUTPUT = 3; // block signal out to TDT
const int PIN_PUMP_OUTPUT = 4; // pump motor
const int PIN_ATTK_OUTPUT = 5; // attack signal out to lobsterbot
const int PIN_TRIAL_INPUT = 6; // door sensor
const int PIN_TRIAL_OUTPUT = 7; // trial signal out to TDT
const int PIN_CLOSE_OUTPUT = 8; // (previously Tone pin) currently using as 
const int PIN_MANUAL_SUC_OUTPUT = 9;
const int PIN_LICK_INPUT = 10;
const int PIN_MANUAL_BUTTON_INPUT = 11;
const int PIN_CLEANUP = 12; // when on, pump and valve is on
const int PIN_PUMP_INPUT = 13;

// Event State Variables
bool isBlockEverStarted = false;
bool isBlockSwitchOn = false;
bool isBlock = false;

bool isDoorClosed = true;
bool isTrial = false;

bool isAttackArmed = false;
bool isAttacked = false;

bool timelimitreached = false;

// Time variable
unsigned long blockOnSetTime = 0;
unsigned long attackOnSetTime = 0;
unsigned long attackOffsetTime;

unsigned long accumLickTime = 0; // accum licking time 
unsigned long lickStartTime = 0;
unsigned long timelimit = 0;     // duration of each Trials


// Etc.
int percentage_attack_in_6sec = 70; //six second attack probability
int trial = 0;
int numLick = 0;

String mode;
bool lickToggle = false;

void setup() 
{
  // Setup : pinMode 
  pinMode(PIN_BLOCK_INPUT, INPUT); //D2
  pinMode(PIN_BLOCK_OUTPUT, OUTPUT); //D3
  pinMode(PIN_PUMP_OUTPUT, OUTPUT); //D4
  pinMode(PIN_ATTK_OUTPUT, OUTPUT); //D5
  pinMode(PIN_TRIAL_INPUT, INPUT); //D6
  pinMode(PIN_TRIAL_OUTPUT, OUTPUT); //D7
  pinMode(PIN_CLOSE_OUTPUT, OUTPUT); //D8
  pinMode(PIN_MANUAL_SUC_OUTPUT, OUTPUT); //D9
  pinMode(PIN_LICK_INPUT, INPUT); //D10
  pinMode(PIN_MANUAL_BUTTON_INPUT, INPUT); //D11
  pinMode(PIN_CLEANUP, INPUT); //D12
  pinMode(PIN_PUMP_INPUT, INPUT); //D13

  // Setup : Setup default output value
  digitalWrite(PIN_BLOCK_OUTPUT,LOW);
  digitalWrite(PIN_PUMP_OUTPUT,LOW);
  digitalWrite(PIN_ATTK_OUTPUT,LOW);
  digitalWrite(PIN_TRIAL_OUTPUT,LOW);
  digitalWrite(PIN_CLOSE_OUTPUT,LOW);
  digitalWrite(PIN_MANUAL_SUC_OUTPUT,LOW);
  
  // Generate random seed
  randomSeed(analogRead(0));

  // Setup : Start Serial Comm
  Serial.begin(9600);
  Serial.println(START_MSG);

  // Setup : Prompt Experiment Mode
  Serial.println("==============Select Mode============");
  Serial.println("+--------------+");
  Serial.println("|      mode    |");
  Serial.println("+---------+----+");
  Serial.println("| train:  | tr |");
  Serial.println("| attack: | at |");
  Serial.println("+---------+----+");
  
  // Setup : Read Experiment Mode
  bool invalidInput = true;
  Serial.println("Mode? : ");
  while (invalidInput)
  {
    if (Serial.available())
    {
      mode = Serial.readString();
      
      if (mode == "at")
      {
        Serial.println("==============Attack Mode============");
        invalidInput = false;
      }
      else if (mode == "tr")
      {
        Serial.println("=============Training Mode===========");
        invalidInput = false;
      }
      else
      {
        Serial.println("Wrong Input");
        Serial.println("Mode? : ");
        invalidInput = true;
      }
    }
  }

  // Setup : Prompt Attack in 6sec percentage
  if (mode == "at")
  {
    Serial.println("=======Select Attack in 6sec %=======");

    // Setup : Read Attack in 6sec percentage
    invalidInput = true;
    Serial.println("Percentage? : ");
    while (invalidInput)
    {
      if (Serial.available())
      {
        percentage_attack_in_6sec = Serial.parseInt();
        if (percentage_attack_in_6sec >= 0 && percentage_attack_in_6sec <=100)
        {
          invalidInput = false;
        }
      }
    }
  }
  else if (mode == "tr")
  {
    Serial.println("No attack : Training Mode");
  }

  // Setup : Time limit
  invalidInput = true;
  Serial.println("Duration? (in minutes): ");
  while (invalidInput)
  {
    if (Serial.available())
    {
      timelimit = Serial.parseInt();
      timelimit = timelimit*1000*60;
      invalidInput = false;
    }
  }


  // Review current protocol
  Serial.println("==========Current Protocol===========");
  if (mode == "at")
  {
    Serial.print("Attack in 6sec : ");
    Serial.println(percentage_attack_in_6sec);
    Serial.print("Attack in 3sec : ");
    Serial.println(100-percentage_attack_in_6sec);
    
  }
  else if (mode == "tr")
  {
    Serial.println("Training Mode");
  }
  Serial.println("=====================================");
}

void loop() 
{
  /*********************/
  /********Block********/
  /*********************/
  isBlockSwitchOn = digitalRead(PIN_BLOCK_INPUT);
  if(!isBlock) // not in block
  {
    if(isBlockSwitchOn == HIGH) // block digital switch on
    {
      digitalWrite(PIN_BLOCK_OUTPUT,HIGH);
      Serial.println("Block started");
      isBlockEverStarted = true;
      blockOnSetTime = millis();
      isBlock = true;
      timelimitreached = false;
      
      // Init. all variables
      numLick = 0;
      trial = 0;
      accumLickTime = 0;
    }
  }
  else // in block
  {
    if(isBlockSwitchOn == LOW && blockOnSetTime + 500 < millis()/* + 500 for jittering*/) // block digital switch off
    {
      digitalWrite(PIN_BLOCK_OUTPUT,LOW);
      Serial.println("Block ended");
      if (isBlockEverStarted)
      {
        Serial.println("##########Experiment Finished##########");
        Serial.print("Number of Total Trials : ");
        Serial.println(trial);
        Serial.print("Number of Total Licks : ");
        Serial.println(numLick);
        Serial.print("Lick Time : ");
        Serial.println(accumLickTime);
        Serial.println("#######################################");
      }
      isBlock = false;
    }
  }

  /*********************/
  /********Trial********/
  /*********************/
  isDoorClosed = digitalRead(PIN_TRIAL_INPUT);
  if(!isDoorClosed) // Door opened = trial started
  {
    if(!isTrial) // Start Trial
    {
      trial = trial+1;
      digitalWrite(PIN_TRIAL_OUTPUT,HIGH);

      if(random(100) < percentage_attack_in_6sec)   //decide long or short attack onset time
        attackOffsetTime = 6000;
      else
        attackOffsetTime = 3000;
      
      // Print Trial Info
      Serial.print("trial : ");
      Serial.println(trial);
      
      Serial.print("Attack in ");
      Serial.print(attackOffsetTime);
      Serial.println("ms sec");
      
      isTrial = true;
    }
    
    if(!isAttacked) // not attacked
    {
      if(!isAttackArmed) // not armed
      {
        if(digitalRead(PIN_LICK_INPUT)) // licked
        {
          isAttackArmed = true;
          attackOnSetTime = millis() + attackOffsetTime;
          Serial.println(" Licked");
        }
      }
      else // armed
      {
        if(attackOnSetTime < millis()) // Attack Onset Time reached
        {
          Serial.println("######Attacked!!######");
          attack();
        }
      }
    }
  }
  else // Door closed
  {
    if(isTrial) // finish trial
    {
      isTrial = false;
      isAttackArmed = false;
      isAttacked = false;
      digitalWrite(PIN_TRIAL_OUTPUT,LOW);
      if (trial%2==0)
      {
        digitalWrite(PIN_PUMP_OUTPUT,HIGH);
        delay(750);
        digitalWrite(PIN_PUMP_OUTPUT,LOW);
      }
    }
  }

  // Manual Button Control and Pump
  if (digitalRead(PIN_CLEANUP) == HIGH)// Clean up ON
  { 
    digitalWrite(PIN_MANUAL_SUC_OUTPUT, HIGH);
    digitalWrite(PIN_PUMP_OUTPUT, HIGH);
  }
  else
  {
  	if (mode == "at")
  	{
  		digitalWrite(PIN_MANUAL_SUC_OUTPUT, LOW);
  		if(digitalRead(PIN_MANUAL_BUTTON_INPUT) == HIGH)
  		{
        Serial.println("Manual Attack");
        attack();
  		}
  	}
    else if (mode == "tr")
    {
    	digitalWrite(PIN_MANUAL_SUC_OUTPUT, digitalRead(PIN_MANUAL_BUTTON_INPUT));
    }
    digitalWrite(PIN_PUMP_OUTPUT,digitalRead(PIN_PUMP_INPUT));
  }
  
  //Num Lick Count
  if(digitalRead(PIN_LICK_INPUT)==HIGH)
  {
    if(!lickToggle)
    {
      lickToggle = true;
      numLick++;

      lickStartTime = millis(); // add lick start time

      if(numLick%10 == 0)
      {
        if(numLick%20 == 0)
        { 
          unsigned long timepassed = millis() - blockOnSetTime;
          Serial.print(long(timepassed / 1000 / 60));
          Serial.print(":");
          Serial.println((timepassed - long(timepassed / 1000 / 60) * 1000 * 60)/1000);
        }
        Serial.println(numLick);
      }
    }
  }
  else
  {
    if(lickToggle)
    {
      lickToggle = false;
      accumLickTime += millis() - lickStartTime;
    }
  }
  //Time alert when set amount of time has passed
  if (timelimitreached == false && millis() - blockOnSetTime>timelimit)
  {
    Serial.println("###### Alert: Timelimit reached! ######");
    timelimitreached = true;
  }
}

void attack()
{
  digitalWrite(PIN_ATTK_OUTPUT,HIGH);
  digitalWrite(PIN_CLOSE_OUTPUT, HIGH); // this sig will command the door controller to close after 1 sec from this sig.
  delay(100);
  digitalWrite(PIN_ATTK_OUTPUT,LOW);
  digitalWrite(PIN_CLOSE_OUTPUT, LOW);
  delay(1000);
  isAttacked = true;
}
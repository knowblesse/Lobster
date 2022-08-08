const int PIN_VALVE_OUTPUT = 5;
const int PIN_LICK_TRIAL_KEY_OUTPUT = 6;
const int PIN_LICK_TDT_OUTPUT = 7;

const int PIN_CONTROL_LICK_INPUT = 8; // previously Valve Inhibit. Manual Button at Control panel
const int PIN_LICKPIN_INPUT = 11;

const int PIN_POWER_ON_OUTPUT = 12;
const int PIN_VALVE_ON_LED_OUTPUT = 13;

bool Lick_trial = false;
bool isLickIRBlocked = true;
bool isManualButtonPushed = false;

void setup() 
{ 
  pinMode(PIN_VALVE_OUTPUT, OUTPUT);
  pinMode(PIN_LICKPIN_INPUT, INPUT);
  pinMode(PIN_LICK_TDT_OUTPUT, OUTPUT); 
  pinMode(PIN_LICK_TRIAL_KEY_OUTPUT, OUTPUT);
  pinMode(PIN_CONTROL_LICK_INPUT, INPUT);
  pinMode(PIN_POWER_ON_OUTPUT, OUTPUT);
  pinMode(PIN_VALVE_ON_LED_OUTPUT,OUTPUT);

  digitalWrite(PIN_POWER_ON_OUTPUT, HIGH);
}


void loop() 
{

  isLickIRBlocked = !digitalRead(PIN_LICKPIN_INPUT);
  isManualButtonPushed = digitalRead(PIN_CONTROL_LICK_INPUT);

  if (isLickIRBlocked) // sensor blocked = licking
  { 
    digitalWrite(PIN_LICK_TDT_OUTPUT, HIGH);
    digitalWrite(PIN_VALVE_OUTPUT, HIGH);
    digitalWrite(PIN_VALVE_ON_LED_OUTPUT,HIGH);
    digitalWrite(PIN_LICK_TRIAL_KEY_OUTPUT,HIGH);
  }
  else // sensor not blocked = not licking
  { 
    digitalWrite(PIN_LICK_TDT_OUTPUT, LOW);
    digitalWrite(PIN_LICK_TRIAL_KEY_OUTPUT, LOW);
    digitalWrite(PIN_VALVE_OUTPUT, LOW);

    if (isManualButtonPushed)// override my manual button
    { 
      digitalWrite(PIN_VALVE_OUTPUT, HIGH);
      digitalWrite(PIN_VALVE_ON_LED_OUTPUT,HIGH);
    }
    else
    { 
      digitalWrite(PIN_VALVE_OUTPUT, LOW);
      digitalWrite(PIN_VALVE_ON_LED_OUTPUT,LOW);
    }
  }
}

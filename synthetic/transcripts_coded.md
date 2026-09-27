# SYNTHETIC transcripts with codes

**Everything here is invented.** See build_synthetic_examples.py.


## SYN-CONV-01 (activity SYN-ACT-03, target DOK 3) — reaches the DOK 3 target on the fourth student turn, then regresses

| Turn | Speaker | Text (synthetic) | Code |
|---|---|---|---|
| 1 | AI | Should the town replace the parking lot with a community garden? Start by telling me the position you lean toward. | socratic_question, request_attempt |
| 2 | Student | I think the garden is better. | DOK 1; explains |
| 3 | AI | Why do you lean that way? What in the readings supports the garden? | socratic_question, request_evidence |
| 4 | Student | The second reading said the town has almost no green space downtown and the lot is only half full most days. | DOK 2; explains, argues_with_evidence |
| 5 | AI | Good, that is evidence. What would someone who wants to keep the parking lot say back to you? | affirm_extend, socratic_question |
| 6 | Student | They would say businesses need the parking. | DOK 2; explains |
| 7 | AI | How would you answer that with what you know from the readings? | socratic_question, request_evidence |
| 8 | Student | The first reading said the lot is half empty most days, so losing it would not really hurt the shops, and the garden would bring more people downtown on weekends, which actually helps the shops more than empty parking spaces do. So the counterargument is weaker than it sounds. | DOK 3; argues_with_evidence, compares_evaluates |
| 9 | AI | You have used evidence to turn the counterargument around. Which of your two points is stronger, and why? | affirm_extend, socratic_question |
| 10 | Student | probably the half empty one | DOK 1; explains |
| 11 | AI | Say more about that: what makes it the stronger point? | socratic_question, request_evidence |
| 12 | Student | because it is a fact from the reading not just an opinion | DOK 2; explains |

## SYN-CONV-02 (activity SYN-ACT-03, target DOK 3) — stays below the DOK 3 target; recall and answer-seeking

| Turn | Speaker | Text (synthetic) | Code |
|---|---|---|---|
| 1 | AI | Should the town replace the parking lot with a community garden? Start by telling me the position you lean toward. | socratic_question, request_attempt |
| 2 | Student | idk | DOK 0; other |
| 3 | AI | That is fine. What is one thing you remember from either reading about the parking lot or the garden? | socratic_question, hint |
| 4 | Student | the garden would have vegetables | DOK 1; explains |
| 5 | AI | Yes. Does that make you lean toward the garden or the parking lot? | affirm_extend, socratic_question |
| 6 | Student | garden | DOK 1; explains |
| 7 | AI | What evidence from the readings supports choosing the garden? | socratic_question, request_evidence |
| 8 | Student | can you just tell me what to write | DOK 0; requests_answer |
| 9 | AI | I cannot write it for you, but I can help you find it. Look at the second reading. What does it say about green space downtown? | redirect, hint, socratic_question |
| 10 | Student | there isnt much green space | DOK 1; explains |
| 11 | AI | Good. Now put those together in one sentence: your claim and that piece of evidence. | affirm_extend, stepwise_guidance, request_attempt |
| 12 | Student | the town should build the garden because there is not much green space downtown | DOK 2; explains, argues_with_evidence |

## SYN-CONV-03 (activity SYN-ACT-02, target DOK 2) — structured task; reaches the DOK 2 target and closes on the deliverable

| Turn | Speaker | Text (synthetic) | Code |
|---|---|---|---|
| 1 | AI | Let's start with step one. Can you name the four stages of the water cycle? | stepwise_guidance, request_attempt |
| 2 | Student | evaporation, condensation, precipitation, collection | DOK 1; explains |
| 3 | AI | All four, nicely done. Step two: describe what happens in evaporation, in one or two sentences. | affirm_extend, stepwise_guidance |
| 4 | Student | The sun heats the ocean and water turns into vapor that rises up. | DOK 2; explains |
| 5 | AI | Right. Now condensation: what happens to that vapor as it rises? | affirm_extend, stepwise_guidance, socratic_question |
| 6 | Student | it gets cold high up so it turns back into tiny drops and makes clouds | DOK 2; explains |
| 7 | AI | Good. Precipitation and collection next: how does the water get from the cloud to a mountain stream? | affirm_extend, stepwise_guidance, socratic_question |
| 8 | Student | the drops get heavy and fall as rain or snow on the mountain, then it runs downhill and collects in streams | DOK 2; explains |
| 9 | AI | You have all four stages in order. Step three: write them as one paragraph. | recap, stepwise_guidance, request_attempt |
| 10 | Student | The water cycle starts when the sun heats the ocean and water evaporates into vapor. The vapor rises, cools, and condenses into clouds. When the drops get heavy they fall as rain or snow on the mountains. The water runs downhill and collects in streams. | DOK 2; explains, submits_work |
| 11 | AI | That paragraph explains all four stages in order. You have completed the task. | affirm_extend, closure |

## SYN-CONV-04 (activity SYN-ACT-01, target DOK 2) — minimal prompt; drifts off task, then a brief DOK 2 explanation

| Turn | Speaker | Text (synthetic) | Code |
|---|---|---|---|
| 1 | AI | Which renewable energy source do you think is most useful? | socratic_question |
| 2 | Student | solar | DOK 1; explains |
| 3 | AI | What makes solar stand out to you compared with the others? | socratic_question |
| 4 | Student | do you know what time lunch is | DOK 0; off_task |
| 5 | AI | I can't help with the schedule, but let's get back to energy. Why solar rather than wind or hydro? | redirect, socratic_question |
| 6 | Student | solar works almost anywhere because most places get sun, but wind only works where it is windy and hydro needs a river | DOK 2; explains, compares_evaluates |
| 7 | AI | That is a real comparison. Is there a downside to solar you would want to mention? | affirm_extend, socratic_question |
| 8 | Student | it doesnt work at night | DOK 1; explains |
| 9 | AI | True. Thanks for sharing your thinking on this. | closure, other |

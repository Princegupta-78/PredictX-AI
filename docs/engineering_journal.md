Q1 What is Remaining Useful Life?
Expected answer

Remaining Useful Life (RUL) is the number of operating cycles an engine can continue running before failure. Predicting RUL allows maintenance teams to repair engines before breakdown while avoiding unnecessary servicing of healthy engines.

Q2 Why is this regression?

Because

Output is

75 cycles

46 cycles

121 cycles

These are continuous numerical values.

Classification would only predict

Fail

Not Fail

which loses useful information.

Q3 Why does engine lifetime vary?

Different engines degrade at different rates due to varying operational conditions and wear patterns. Therefore, identical maintenance schedules would either waste maintenance on healthy engines or miss engines that deteriorate faster.

Q4 Why are some sensors useless?

Because

Some sensors remain almost constant throughout the engine's life and therefore carry no information about degradation.

These sensors add noise without improving prediction accuracy.





# Day 2: Exploratory Data Analysis & Target Generation

## Dataset Overview
- **Shape:** 20631 rows, 26 columns
- **Missing Values:** None
- **Target Variable (RUL):** Successfully computed. RUL ranges from 0 (failure) to 361 cycles. Mean RUL is 107.8.

## Sensor Analysis
- **Dead Sensors (To Drop):** Sensors 1, 5, 10, 16, 18, and 19 exhibit zero variance (flatlines). They return NaN correlation and provide no predictive signal.
- **Strong Signals:** Sensors 11, 4, 15, and 17 strongly negatively correlate with RUL. Sensors 12, 7, 21, and 20 strongly positively correlate.
- **Operating Conditions:** In FD001, op_setting_1 and op_setting_2 exhibit microscopic variance. op_setting_3 is constant at 100.0. 

## Next Steps
For Day 3, we will drop the 6 zero-variance sensors and the 3 operating settings to reduce noise. We will then engineer rolling features (moving averages, standard deviations) on the highly correlated sensors to capture the degradation trends over time before training the baseline model.
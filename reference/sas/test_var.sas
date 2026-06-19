/******************************************************************************
 Test driver for the SAS VaR macros.

 Loads the fixture data, runs each method, and prints results for manual
 comparison against the expected_*.csv files.

 Usage:
   1. Update fixtures_path below to the absolute path of ../fixtures/
   2. %include "var.sas" (or paste it into your session)
   3. Run this script
   4. Compare printed output to expected_historical.csv, expected_parametric.csv
******************************************************************************/

/* UPDATE THIS PATH to your local fixtures directory. */
%let fixtures_path = /path/to/reference/fixtures;

%include "var.sas";

/* Load fixture returns. */
proc import datafile="&fixtures_path./sample_returns.csv"
    out=sample_returns dbms=csv replace;
    getnames=yes;
run;

proc sql noprint;
    select count(*) into :n_returns from sample_returns;
quit;
%put NOTE: Loaded &n_returns. returns;

/* ---------------------------------------------------------------------------
   Historical VaR at each confidence/horizon combo
   --------------------------------------------------------------------------- */
%put NOTE: --- Historical Simulation ---;

%macro run_historical;
    %let confs = 0.90 0.95 0.99;
    %let hps = 1 5 10;

    %do c = 1 %to 3;
        %let conf = %scan(&confs., &c.);
        %do h = 1 %to 3;
            %let hp = %scan(&hps., &h.);
            %var_historical(data=sample_returns, confidence=&conf., holding_period=&hp.,
                            out_var=hist_var, out_cvar=hist_cvar);
            %put NOTE: conf=&conf. hp=&hp. VaR=&hist_var. CVaR=&hist_cvar.;
        %end;
    %end;
%mend;
%run_historical;

/* ---------------------------------------------------------------------------
   Parametric VaR at each confidence/horizon combo
   --------------------------------------------------------------------------- */
%put NOTE: --- Parametric (Variance-Covariance) ---;

%macro run_parametric;
    %let confs = 0.90 0.95 0.99;
    %let hps = 1 5 10;

    %do c = 1 %to 3;
        %let conf = %scan(&confs., &c.);
        %do h = 1 %to 3;
            %let hp = %scan(&hps., &h.);
            %var_parametric(data=sample_returns, confidence=&conf., holding_period=&hp.,
                            out_var=param_var, out_cvar=param_cvar);
            %put NOTE: conf=&conf. hp=&hp. VaR=&param_var. CVaR=&param_cvar.;
        %end;
    %end;
%mend;
%run_parametric;

/* ---------------------------------------------------------------------------
   Monte Carlo VaR (95% confidence, 1-day, for a quick check)
   --------------------------------------------------------------------------- */
%put NOTE: --- Monte Carlo (95%% 1-day) ---;
%var_monte_carlo(data=sample_returns, confidence=0.95, holding_period=1,
                 n_simulations=10000, seed=42, out_var=mc_var, out_cvar=mc_cvar);
%put NOTE: VaR=&mc_var. CVaR=&mc_cvar.;

%put NOTE: Done. Compare output above to expected_*.csv fixtures.;

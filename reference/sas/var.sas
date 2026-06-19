/******************************************************************************
 VaR and Expected Shortfall (CVaR) - SAS macro reference implementation.

 Three methods:
   %var_historical   - empirical percentile
   %var_parametric   - closed-form under normality
   %var_monte_carlo  - GBM simulation

 Each macro reads a dataset of returns (single column: return) and writes
 results to macro variables. See test_var.sas for usage.

 No SAS/STAT or SAS/ETS required - base SAS only.
******************************************************************************/

%macro var_historical(data=, confidence=0.95, holding_period=1, out_var=, out_cvar=);
    /* Sort returns, read off the percentile. */
    proc univariate data=&data. noprint;
        var return;
        output out=_var_hist_pctl_
            pctlpts=%sysevalf((1 - &confidence.) * 100)
            pctlpre=p;
    run;

    /* CVaR: average of returns at or below the threshold. */
    data _null_;
        set _var_hist_pctl_;
        call symputx('_var_hist_threshold', p%sysevalf((1 - &confidence.) * 100, integer));
    run;

    proc sql noprint;
        select -mean(return) * sqrt(&holding_period.)
            into :&out_cvar.
            from &data.
            where return <= &_var_hist_threshold.;

        select -&_var_hist_threshold. * sqrt(&holding_period.)
            into :&out_var.
            from _var_hist_pctl_;
    quit;

    proc delete data=_var_hist_pctl_; run;
%mend var_historical;


%macro var_parametric(data=, confidence=0.95, holding_period=1, out_var=, out_cvar=);
    /* Closed-form VaR/CVaR assuming returns ~ N(mu, sigma^2). */
    proc sql noprint;
        select mean(return), std(return)
            into :_vp_mu, :_vp_sigma
            from &data.;
    quit;

    data _null_;
        z = probit(&confidence.);
        mu = &_vp_mu.;
        sigma = &_vp_sigma.;
        h = &holding_period.;
        alpha = 1 - &confidence.;

        var_val = -(mu * h - z * sigma * sqrt(h));
        cvar_val = -(mu * h - sigma * sqrt(h) * pdf('NORMAL', z) / alpha);

        call symputx("&out_var.", var_val);
        call symputx("&out_cvar.", cvar_val);
    run;
%mend var_parametric;


%macro var_monte_carlo(data=, confidence=0.95, holding_period=1,
                       n_simulations=10000, seed=42, out_var=, out_cvar=);
    /* Estimate params from data, simulate GBM, read off percentile. */
    proc sql noprint;
        select mean(return), std(return)
            into :_vmc_mu, :_vmc_sigma
            from &data.;
    quit;

    data _var_mc_sims_;
        call streaminit(&seed.);
        mu = &_vmc_mu.;
        sigma = &_vmc_sigma.;
        dt = &holding_period.;
        drift = (mu - 0.5 * sigma**2) * dt;
        diffusion = sigma * sqrt(dt);
        do i = 1 to &n_simulations.;
            z = rand('NORMAL');
            return = drift + diffusion * z;
            output;
        end;
        keep return;
    run;

    proc univariate data=_var_mc_sims_ noprint;
        var return;
        output out=_var_mc_pctl_
            pctlpts=%sysevalf((1 - &confidence.) * 100)
            pctlpre=p;
    run;

    data _null_;
        set _var_mc_pctl_;
        call symputx('_var_mc_threshold', p%sysevalf((1 - &confidence.) * 100, integer));
    run;

    proc sql noprint;
        select -mean(return)
            into :&out_cvar.
            from _var_mc_sims_
            where return <= &_var_mc_threshold.;

        select -&_var_mc_threshold.
            into :&out_var.
            from _var_mc_pctl_;
    quit;

    proc delete data=_var_mc_sims_ _var_mc_pctl_; run;
%mend var_monte_carlo;

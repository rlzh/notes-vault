#general #security

## CVSS (Common Vulnerability Scoring System) v2.0 Equations
### CVSS Base Score Equation

BaseScore = (.6*Impact +.4*Exploitability-1.5)*f(Impact)
 
Impact = 10.41 * (1 - (1 - ConfImpact) * (1 - IntegImpact) * (1 - AvailImpact))

Exploitability = 20 * AccessComplexity * Authentication * AccessVector

f(Impact) = 0 if Impact=0; 1.176 otherwise
 
AccessComplexity = case AccessComplexity of
                        high:   0.35
                        medium: 0.61
                        low:    0.71
 
Authentication   = case Authentication of
                        Requires no authentication:                    0.704
                        Requires single instance of authentication:    0.56
                        Requires multiple instances of authentication: 0.45
 
AccessVector     = case AccessVector of
                        Requires local access:    0.395
                        Local Network accessible: 0.646
                        Network accessible:       1
 
ConfImpact       = case ConfidentialityImpact of
                        none:             0
                        partial:          0.275
                        complete:         0.660
 
IntegImpact      = case IntegrityImpact of
                        none:             0
                        partial:          0.275
                        complete:         0.660
 
AvailImpact      = case AvailabilityImpact of
                        none:             0
                        partial:          0.275
                        complete:         0.660
 
### CVSS Temporal Equation
 
TemporalScore = BaseScore 
              * Exploitability 
              * RemediationLevel 
              * ReportConfidence
 
Exploitability   = case Exploitability of
                        unproven:             0.85
                        proof-of-concept:     0.9
                        functional:           0.95
                        high:                 1.00
                        not defined           1.00
                        
RemediationLevel = case RemediationLevel of
                        official-fix:         0.87
                        temporary-fix:        0.90
                        workaround:           0.95
                        unavailable:          1.00
                        not defined           1.00
 
ReportConfidence = case ReportConfidence of
                        unconfirmed:          0.90
                        uncorroborated:       0.95      
                        confirmed:            1.00
                        not defined           1.00
 
### CVSS Environmental Equation
 
EnvironmentalScore = (AdjustedTemporal 
                        + (10 - AdjustedTemporal) 
                        * CollateralDamagePotential) 
                     * TargetDistribution
 
AdjustedTemporal = TemporalScore recomputed with the Impact sub-equation 
                   replaced with the following AdjustedImpact equation.
 
AdjustedImpact = Min(10, 
                     10.41 * (1 - 
                                (1 - ConfImpact * ConfReq) 
                              * (1 - IntegImpact * IntegReq) 
                              * (1 - AvailImpact * AvailReq)))
 
CollateralDamagePotential = case CollateralDamagePotential of
                                 none:            0
                                 low:             0.1
                                 low-medium:      0.3   
                                 medium-high:     0.4
                                 high:            0.5      
                                 not defined:     0
                                 
TargetDistribution        = case TargetDistribution of
                                 none:            0
                                 low:             0.25
                                 medium:          0.75
                                 high:            1.00
                                 not defined:     1.00
 
ConfReq       = case ConfidentialityImpact of
                        Low:              0.5
                        Medium:           1
                        High:             1.51
                        Not defined       1
 
IntegReq      = case IntegrityImpact of
                        Low:              0.5
                        Medium:           1
                        High:             1.51
                        Not defined       1
 
AvailReq      = case AvailabilityImpact of
                        Low:              0.5
                        Medium:           1
                        High:             1.51
                        Not defined       1




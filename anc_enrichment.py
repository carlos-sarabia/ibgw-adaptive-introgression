import sys
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multitest as smt

def extract_columns(filename):
    df = pd.read_csv(filename, delimiter='\t', header=None, usecols=[0, 1, 2])
    return df

def calculate_p_value(chisq_value):
    p_value = 1 - stats.chi2.cdf(chisq_value, df=1)
    return p_value

def main():
    if len(sys.argv) != 8:
        print("Usage: python script.py obs_DOG_alt obs_NEGW_alt exp_ref exp_alt obs_ref obs_alt output_file")
        sys.exit(1)

    file1, file2, file3, file4, file5, file6, output_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]

    # Extract columns from each file
    df1 = extract_columns(file1)
    df2 = extract_columns(file2)
    df3 = extract_columns(file3)
    df4 = extract_columns(file4)
    df5 = extract_columns(file5)
    df6 = extract_columns(file6)

    # Store columns in a table
    table = pd.concat([df1, df2.iloc[:, 2], df3.iloc[:, 2], df4.iloc[:, 2], df5.iloc[:, 2], df6.iloc[:, 2]], axis=1)
    table.columns = ['chr', 'pos', 'obs_DOG_alt', 'obs_NEGW_alt', 'exp_ref', 'exp_alt', 'obs_IBGW_ref', 'obs_IBGW_alt']

    # Filter out rows where exp_ref or exp_alt are 0
    table = table[(table['exp_ref'] != 0) & (table['exp_alt'] != 0)]

    # Calculate the chi-squared values and add them as a new column 'chisq'
    table['chisq'] = (((table['obs_IBGW_ref'] - table['exp_ref']) ** 2) / table['exp_ref']) + (((table['obs_IBGW_alt'] - table['exp_alt']) ** 2) / table['exp_alt'])

    # Calculate p-values for each chi-squared value and add them as a new column 'p_val'
    table['p_val'] = table['chisq'].apply(calculate_p_value)

    # Perform False Discovery Rate (FDR) correction on the p-values and add them as a new column 'q_val'
    _, table['q_val'], _, _ = smt.multipletests(table['p_val'], method='fdr_bh')
    
    # Estimate components
    table['DOG_contr'] =  ((table['obs_DOG_alt']-table['obs_NEGW_alt'])*2/(2-abs(table['obs_IBGW_alt']-table['obs_DOG_alt'])))
    table['NEGW_contr'] =  ((table['obs_NEGW_alt']-table['obs_DOG_alt'])*2/(2-abs(table['obs_IBGW_alt']-table['obs_NEGW_alt'])))

    # Save the table to the output file
    table.to_csv(output_file, sep='\t', index=False)

    print(f"Table with chi-squared values and q-values saved to {output_file}")

if __name__ == "__main__":
    main()

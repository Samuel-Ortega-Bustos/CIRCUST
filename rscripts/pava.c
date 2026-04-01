#include <stdio.h>
#include <stdlib.h>

int max(int a, int b)			
{								
	return (a > b ? a : b);		
}				


void pavaCAdri (int *nin, double *y, double *w, double *resultado){

	//Seccion de declaraciones
    int n = nin[0];
    int i, j, k, l;
    double *wm, *ym;
    int *s;

  	//Reserva de espacio
  	wm = (double*) malloc(n * sizeof(double));
  	ym = (double*) malloc(n * sizeof(double));
  	s = (int*) malloc((n+1) * sizeof(int));

    //Inicialización del algoritmo
    //ym contendrá los valores de y modificados según el PAVA
    //wm contendrá los valores de w modificados según el PAVA
    ym[0] = y[0];
    wm[0] = w[0];

    //j indica el número de valores distintos de la regresion isotonica
    j = 1;

    //S[k] sirve para indicar donde comienza la secuencia de valores
    //que toman el valor ym[k-1]
    s[0] = 0;
    s[1] = 1;

    //Comenzamos por el segundo valor
    

    for(i = 1; i < n; i++){

        //Siguiente valor a considerar
        ym[j] = y[i];
        wm[j] = w[i];

        //Si el nuevo valor viola las restricciones de orden, se combina
        //con los anteriores hasta que esto no ocurra
        
        
        while((j >= 1) && (ym[j] <= ym[max(j-1,0)])){
            ym[j-1] = (wm[j]*ym[j] + wm[j-1]*ym[j-1])/(wm[j] + wm[j-1]);
            wm[j-1] = wm[j] + wm[j-1];
            j--;
        }
		

        j++;

        //El siguiente valor comenzaría en el lugar i+1
        s[j] = i+1;

    }

 
    //Para todos los posibles valores existentes, se añaden los 
    //resultados al vector final
    
    for(k = 1; k <= j; k++){
        for(l = s[k-1]; l < s[k]; l++){
            resultado[l] = ym[k-1];
        }
    }

    free(wm);
    free(ym);
    free(s);

}	





void pavaCAdriDecreasing (int *nin, double *y, double *w, double *resultado){

	//Seccion de declaraciones
    int n = nin[0];
    int i, j, k, l;
    double *wm, *ym;
    int *s;

  	//Reserva de espacio
  	wm = (double*) malloc(n * sizeof(double));
  	ym = (double*) malloc(n * sizeof(double));
  	s = (int*) malloc((n+1) * sizeof(int));

    //Inicialización del algoritmo
    //ym contendrá los valores de y modificados según el PAVA
    //wm contendrá los valores de w modificados según el PAVA
    ym[0] = y[0];
    wm[0] = w[0];

    //j indica el número de valores distintos de la regresion isotonica
    j = 1;

    //S[k] sirve para indicar donde comienza la secuencia de valores
    //que toman el valor ym[k-1]
    s[0] = 0;
    s[1] = 1;

    //Comenzamos por el segundo valor
    

    for(i = 1; i < n; i++){

        //Siguiente valor a considerar
        ym[j] = y[i];
        wm[j] = w[i];

        //Si el nuevo valor viola las restricciones de orden, se combina
        //con los anteriores hasta que esto no ocurra
        
        
        while((j >= 1) && (ym[j] >= ym[max(j-1,0)])){
            ym[j-1] = (wm[j]*ym[j] + wm[j-1]*ym[j-1])/(wm[j] + wm[j-1]);
            wm[j-1] = wm[j] + wm[j-1];
            j--;
        }
		

        j++;

        //El siguiente valor comenzaría en el lugar i+1
        s[j] = i+1;

    }

 
    //Para todos los posibles valores existentes, se añaden los 
    //resultados al vector final
    
    for(k = 1; k <= j; k++){
        for(l = s[k-1]; l < s[k]; l++){
            resultado[l] = ym[k-1];
        }
    }

    free(wm);
    free(ym);
    free(s);

}
								
				
								


#include <stdio.h>
#include <stdlib.h>

int max(int a, int b)			
{								
	return (a > b ? a : b);		
}				
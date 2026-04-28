#include <R.h>
#include <Rinternals.h>
#include <libintl.h>
#include <stdio.h>
#include <string.h>
#include <float.h>

//Copia el contenido del vector original a su copia
void copiar(int n, double *original, double*copia){
    int i;
    for(i = 0; i < n; i++){
        copia[i] = original[i];
    }
}

void reverse(int n, int *original, int *revertido){
    int i;
    for(i = 0; i < n; i++){
        revertido[n-i-1] = original[i];
    }
}

double mean(int m, const double *a) {
    double sum; 
    int i;
    sum = 0;
    for(i=0; i<m; i++)
        sum+=a[i];
    return(sum/m);
}

double calculoMSE(int n, const double *original, double *ajuste){
    double resultado, resta;
    int i;
    resultado = 0;
    for(i = 0; i < n; i++){
        resta = original[i] - ajuste[i];
        resultado += resta*resta;
    }
    return(resultado/n);
}

//Reordena el vector original, de manera que los primeros elementos que se colocan
//son los mayores que indL, y después van el resto
void encontrarOrdenComprobacion(int indL, int n, const int *original, int *reordenado){
    int i, j, cond;
    
    //Encontrar a partir de cuándo se cumple la condición
    cond = 0;
    for(i = 0; i < n; i++){
        if(original[i] >= indL){
            cond = i;
            break;
        }
    }

    //Reordenar, eligiendo primero a partir de los que se cumple la condición
    j = 0;
    for(i = cond; i < n ; i++){
        reordenado[j] = original[i];
        j++;
    }
    for(i = 0; i < cond; i++){
        reordenado[j] = original[i];
        j++;
    }
}




//Calcula el pava creciente para el vector y, desde la posición marcada y hasta
//la posición marcada. El resultado se almacena en la dirección proporionada
void pavaCreciente(double *original, int desde, int hasta, double *resultado) {

    //Seccion de declaraciones
    int n = hasta - desde + 1;
    int i, j, k, l;
    double *ym, *wm, *w, *y;
    int *s;

    //Reserva de espacio para estructuras
    ym = (double *) malloc (n * sizeof(double));
    wm = (double *) malloc (n * sizeof(double));
    w = (double *) malloc (n * sizeof(double));
    s = (int *) malloc ((n+1) * sizeof(int));


    //Inicialización del algoritmo
    //ym contendrá los valores de y modificados según el PAVA
    //wm contendrá los valores de w modificados según el PAVA
    y = original + desde;
    for(i = 0; i < n; i++){
        w[i] = 1;
    }
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
        while((j > 0) && (ym[j] < ym[j-1])){
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

    free(ym);
    free(wm);
    free(w);
    free(s);

}

//Calcula el pava decreciente para el vector y, desde la posición marcada y hasta
//la posición marcada. El resultado se almacena en la dirección proporionada
void pavaDecreciente(double *original, int desde, int hasta, double *resultado) {

    //Seccion de declaraciones
    int n = hasta - desde + 1;
    int i, j, k, l;
    double *ym, *wm, *w, *y;
    int *s;

    //Reserva de espacio para estructuras
    ym = (double *) malloc (n * sizeof(double));
    wm = (double *) malloc (n * sizeof(double));
    w = (double *) malloc (n * sizeof(double));
    s = (int *) malloc ((n+1) * sizeof(int));


    //Inicialización del algoritmo
    //ym contendrá los valores de y modificados según el PAVA
    //wm contendrá los valores de w modificados según el PAVA
    y = original + desde;
    for(i = 0; i < n; i++){
        w[i] = 1;
    }
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
        while((j > 0) && (ym[j] > ym[j-1])){
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

    free(ym);
    free(wm);
    free(w);
    free(s);

}


SEXP busquedaMejor(SEXP v, SEXP candL, SEXP candU) {

    //Sección de declaraciones
    int lengthV, lengthCandL, lengthCandU, nCandUValidos, lengthPavaLU, lengthPavaUL;
    int i, j, iL, iU, indL, indU;
    SEXP pavaFin, mseFin, Lopt, Uopt, ans;
    double media, mseAux;
    double *v2, *pavaLU, *pavaUL, *pavaLUrecuperado, *pavaAux;
    double **pavasCrecientes;
    int *ordenComprobacion, *candUValidos, *candUValidosreversed, *lengthPavasCrecientes;

    //Inicialización de variables
    lengthV = LENGTH(v);
    lengthCandU = LENGTH(candU);
    lengthCandL = LENGTH(candL);

    //Reserva de espacio para estructuras internas
    v2 = (double *) malloc( 2*lengthV*sizeof(double));
    pavaLU = (double *) malloc (lengthV * sizeof(double));
    pavaUL = (double *) malloc (lengthV * sizeof(double));
    pavaAux = (double *) malloc (lengthV * sizeof(double));
    ordenComprobacion = (int *) malloc(lengthCandU * sizeof(int));
    candUValidos = (int *) malloc(lengthCandU * sizeof(int));
    candUValidosreversed = (int *) malloc(lengthCandU * sizeof(int));
    lengthPavasCrecientes = (int *) malloc(lengthCandU * sizeof(int));
    pavasCrecientes = (double **) malloc(lengthCandU * sizeof(double *));

    //Reserva de espacio para estructuras que se van a devolver
    const char *anms[] = {"v", "candL", "candU", "pavaFin", "mseFin", "Lopt", "Uopt", ""};

    PROTECT(ans = mkNamed(VECSXP, anms));

    SET_VECTOR_ELT(ans, 0, v);
    SET_VECTOR_ELT(ans, 1, candL);
    SET_VECTOR_ELT(ans, 2, candU);
    SET_VECTOR_ELT(ans, 3, pavaFin = allocVector(REALSXP, lengthV));
    SET_VECTOR_ELT(ans, 4, mseFin = allocVector(REALSXP, 1));
    SET_VECTOR_ELT(ans, 5, Lopt = allocVector(INTSXP, 1));
    SET_VECTOR_ELT(ans, 6, Uopt = allocVector(INTSXP, 1));

    //Inicialización de datos de las estructuras
    for(i = 0; i < lengthV; i++){
        v2[i] = v2[i + lengthV] = REAL(v)[i];
    }
    REAL(mseFin)[0] = DBL_MAX;
    INTEGER(Lopt)[0] = -1;
    INTEGER(Uopt)[0] = -1;

    //Inicio del algoritmo

    //Para cada combinación de mínimo local (indL) y máximo local (indU)
    //hay que hacer el ajuste
    for(iL = 0; iL < lengthCandL; iL++){
        indL = INTEGER(candL)[iL]; //En C la primera posición de un vector es la 0

        //Se modifica el orden en el que se comprobarán los máximos,
        //haciéndolo a partir de la derecha de indL
        encontrarOrdenComprobacion(indL,lengthCandU,INTEGER(candU),ordenComprobacion);
        
        //Se inicializa el número de candidatos válidos a 0.
        //No hace falta reiniciar el vector candUValidos, pues se irán modificando
        //las posiciones de memoria que vayan siendo necesarias. El número de 
        //candidatos válidos indicará hasta qué valor se puede comprobar.
        //Los pavas crecientes se liberan al final de la ejecución anterior. Cuando
        //sea necesario guardar un pava creciente, hay que reservar memoria
        nCandUValidos = 0;

        ////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////
        //////////// PRIMERA PARTE: CÁLCULO DE LOS PAVAS CRECIENTES ////////
        ////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////

        for(iU = 0; iU < lengthCandU; iU++){
            indU = ordenComprobacion[iU];

            lengthPavaLU = 0;

            ///////////////////////////////////////////////////////
            //////// PRIMER CASO: L SE PRODUCE ANTES QUE U ////////
            ///////////////////////////////////////////////////////
            if(indL < indU){

                //Cálculo del PAVA creciente
                lengthPavaLU = indU - indL + 1;
                if(lengthPavaLU > 2){
                    pavaCreciente(REAL(v),indL+1,indU-1,pavaLU+1);
                }
                pavaLU[0] = REAL(v)[indL];
                pavaLU[lengthPavaLU-1] = REAL(v)[indU];

                //Si el ajuste no es adecuado por la derecha del mínimo local,
                //no hace falta seguir probando con más U's, puesto que el inicio
                //del PAVA no se verá modificado
                if(pavaLU[1] <= REAL(v)[indL]){
                    break;
                }

                //Para saber si indU es un candidato válido, se debe cumplir la
                //condición por la izquierda de indU: el máximo local es mayor que el
                //inmediato anterior. Si es válido, se guarda el pava creciente usado
                if(pavaLU[lengthPavaLU-2] < REAL(v)[indU]){
                    candUValidos[nCandUValidos] = indU;
                    pavasCrecientes[nCandUValidos] = (double *) malloc (lengthPavaLU * sizeof(double));
                    copiar(lengthPavaLU,pavaLU,pavasCrecientes[nCandUValidos]);
                    lengthPavasCrecientes[nCandUValidos] = lengthPavaLU;
                    nCandUValidos++;
                }

            ///////////////////////////////////////////////////////
            //////// SEGUNDO CASO: L Y U SON LOS MISMOS ///////////
            ///////////////////////////////////////////////////////            
            } else if(indL == indU){

                //En este caso el pava siempre será válido, y es constante
                media = mean(lengthV,REAL(v));
                for(i = 0; i < lengthV; i++){
                    pavaLU[i] = media;
                }
                lengthPavaLU = lengthV;
                candUValidos[nCandUValidos] = indU;
                pavasCrecientes[nCandUValidos] = (double *) malloc (lengthPavaLU * sizeof(double));
                copiar(lengthPavaLU,pavaLU,pavasCrecientes[nCandUValidos]);
                lengthPavasCrecientes[nCandUValidos] = lengthPavaLU;
                nCandUValidos++;

            ///////////////////////////////////////////////////////
            //////// TERCER CASO: L SE PRODUCE DESPUÉS DE U ///////
            ///////////////////////////////////////////////////////  
            } else {

                //Cálculo del PAVA creciente
                lengthPavaLU = lengthV - indL + 1 + indU;
                if(lengthPavaLU > 2){
                    pavaCreciente(v2,indL+1,lengthV + indU - 1,pavaLU+1);
                }
                pavaLU[0] = REAL(v)[indL];
                pavaLU[lengthPavaLU-1] = REAL(v)[indU];

                //Si el ajuste no es adecuado por la derecha del mínimo local,
                //no hace falta seguir probando con más U's, puesto que el inicio
                //del PAVA no se verá modificado
                if(pavaLU[1] <= REAL(v)[indL]){
                    break;
                }

                //Para saber si indU es un candidato válido, se debe cumplir la
                //condición por la izquierda de indU: el máximo local es mayor que el
                //inmediato anterior. Si es válido, se guarda el pava creciente usado
                if(pavaLU[lengthPavaLU-2] < REAL(v)[indU]){
                    candUValidos[nCandUValidos] = indU;
                    pavasCrecientes[nCandUValidos] = (double *) malloc (lengthPavaLU * sizeof(double));
                    copiar(lengthPavaLU,pavaLU,pavasCrecientes[nCandUValidos]);
                    lengthPavasCrecientes[nCandUValidos] = lengthPavaLU;
                    nCandUValidos++;
                }
            }
        }

        ////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////
        ////////// SEGUNDA PARTE: CÁLCULO DE LOS PAVAS DECRECIENTES ////////
        ////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////
        reverse(nCandUValidos, candUValidos, candUValidosreversed);

        //Exploramos los candidatos válidos al revés. La idea es la misma que
        //la anterior: si no se cumple la condición por la izquierda de indL
        //a partir de un cierto indU, entonces añadir más puntos no lo modificará
        for(iU = 0; iU < nCandUValidos; iU++){
            indU = candUValidosreversed[iU];
            lengthPavaUL = 0;

            //El pava creciente correspondiente es:
            pavaLUrecuperado = pavasCrecientes[nCandUValidos - iU - 1];
            lengthPavaLU = lengthPavasCrecientes[nCandUValidos - iU - 1];

            ///////////////////////////////////////////////////////
            //////// PRIMER CASO: L SE PRODUCE ANTES QUE U ////////
            ///////////////////////////////////////////////////////
            if(indL < indU){

                //Si el pava creciente no era desde el primer hasta el último
                //punto, hay que construir pavaUL en dos partes
                if(indU != lengthV-1 && indL != 0){

                    //Cálculo del PAVA decreciente
                    lengthPavaUL = lengthV - indU - 1 + indL;
                    pavaDecreciente(v2, indU+1, lengthV+indL-1,pavaUL);

                    //Si no se cumple la condición izquierda de indL, ningún U
                    //a la izquierda será válido
                    if(pavaUL[lengthPavaUL-1] < REAL(v)[indL]){
                        break;
                    }

                    //Si es válido, se calcula el estimador conjunto y su MSE
                    //Se tomará como solución actual si el MSE es el mínimo encontrado
                    if(pavaUL[0] <= REAL(v)[indU]){

                        //Ajuste completo
                        j = 0;
                        for(i = lengthPavaUL - indL; i < lengthPavaUL; i++){
                            pavaAux[j] = pavaUL[i];
                            j++;
                        }
                        for(i = 0; i < lengthPavaLU; i++){
                            pavaAux[j] = pavaLUrecuperado[i];
                            j++;
                        }
                        for(i = 0; i < lengthPavaUL - indL; i++){
                            pavaAux[j] = pavaUL[i];
                            j++;
                        }
                        mseAux = calculoMSE(lengthV, REAL(v), pavaAux);

                        if(mseAux < REAL(mseFin)[0]){
                            REAL(mseFin)[0] = mseAux;
                            INTEGER(Lopt)[0] = indL+ 1;
                            INTEGER(Uopt)[0] = indU + 1;
                            for(i = 0; i < lengthV; i++){
                                REAL(pavaFin)[i] = pavaAux[i];
                            }
                        }
                    }

                //En este caso se considera el PAVA donde indU es el último punto
                } else if (indU == lengthV-1 && indL != 0) {

                    //Cálculo del PAVA decreciente
                    lengthPavaUL = indL;
                    pavaDecreciente(REAL(v), 0, indL-1, pavaUL);

                    //Si no se cumple la condición por la izquierda, no se continúa
                    if(pavaUL[lengthPavaUL-1] < REAL(v)[indL]){
                        break;
                    }

                    //Si es válido, se calcula el MSE
                    if(pavaUL[0] <= REAL(v)[indU]){

                        //Ajuste completo
                        j = 0;
                        for(i = 0; i < lengthPavaUL; i++){
                            pavaAux[j] = pavaUL[i];
                            j++;
                        }
                        for(i = 0; i < lengthPavaLU; i++){
                            pavaAux[j] = pavaLUrecuperado[i];
                            j++;
                        }
                        mseAux = calculoMSE(lengthV, REAL(v), pavaAux);    

                        if(mseAux < REAL(mseFin)[0]){
                            REAL(mseFin)[0] = mseAux;
                            INTEGER(Lopt)[0] = indL+ 1;
                            INTEGER(Uopt)[0] = indU + 1;
                            for(i = 0; i < lengthV; i++){
                                REAL(pavaFin)[i] = pavaAux[i];
                            }
                        }
                    }

                //En este caso se considera el PAVA donde L es el primer punto
                } else if (indL == 0 && indU != lengthV-1){

                    //Cálculo del PAVA decreciente
                    lengthPavaUL = lengthV - indU - 1;
                    pavaDecreciente(REAL(v), indU+1, lengthV-1, pavaUL);

                    //Si no se cumple la condición por la izquierda, no se continúa
                    if(pavaUL[lengthPavaUL-1] < REAL(v)[indL]){
                        break;
                    }

                    //Si es válido, se calcula el MSE
                    if(pavaUL[0] <= REAL(v)[indU]){

                        //Ajuste completo
                        j = 0;
                        for(i = 0; i < lengthPavaLU; i++){
                            pavaAux[j] = pavaLUrecuperado[i];
                            j++;
                        }
                        for(i = 0; i < lengthPavaUL; i++){
                            pavaAux[j] = pavaUL[i];
                            j++;
                        }
                        mseAux = calculoMSE(lengthV, REAL(v), pavaAux);    

                        if(mseAux < REAL(mseFin)[0]){
                            REAL(mseFin)[0] = mseAux;
                            INTEGER(Lopt)[0] = indL+ 1;
                            INTEGER(Uopt)[0] = indU + 1;
                            for(i = 0; i < lengthV; i++){
                                REAL(pavaFin)[i] = pavaAux[i];
                            }
                        }
                    }

                //En este caso se considera el PAVA cuando L es el mínimo y U el máximo
                } else {

                    //No hay que hacer PAVA decreciente, sólo está el creciente                  
                    mseAux = calculoMSE(lengthV, REAL(v), pavaLUrecuperado); 

                    if(mseAux < REAL(mseFin)[0]){
                        REAL(mseFin)[0] = mseAux;
                        INTEGER(Lopt)[0] = indL+ 1;
                        INTEGER(Uopt)[0] = indU + 1;
                        for(i = 0; i < lengthV; i++){
                            REAL(pavaFin)[i] = pavaLUrecuperado[i];
                        }
                    }                    
                }

            ///////////////////////////////////////////////////////
            //////// SEGUNDO CASO: L Y U SON LOS MISMOS ///////////
            ///////////////////////////////////////////////////////  
            } else if (indL == indU){

                //El estimador es constante
                mseAux = calculoMSE(lengthV, REAL(v), pavaLUrecuperado); 

                if(mseAux < REAL(mseFin)[0]){
                    REAL(mseFin)[0] = mseAux;
                    INTEGER(Lopt)[0] = indL+ 1;
                    INTEGER(Uopt)[0] = indU + 1;
                    for(i = 0; i < lengthV; i++){
                        REAL(pavaFin)[i] = pavaLUrecuperado[i];
                    }
                } 


            ///////////////////////////////////////////////////////
            //////// TERCER CASO: L SE PRODUCE DESPUÉS DE U ///////
            ///////////////////////////////////////////////////////
            } else {

                //Si el PAVA creciente abarca todos los puntos, no es necesario
                //el decreciente
                if(lengthPavaLU == lengthV){

                    j = 0;
                    for(i = lengthV - indL; i < lengthV; i++){
                        pavaAux[j] = pavaLUrecuperado[i];
                        j++;
                    }
                    for(i = 0; i < lengthV - indL; i++){
                        pavaAux[j] = pavaLUrecuperado[i];
                        j++;                        
                    }

                    mseAux = calculoMSE(lengthV, REAL(v), pavaAux); 

                    if(mseAux < REAL(mseFin)[0]){
                        REAL(mseFin)[0] = mseAux;
                        INTEGER(Lopt)[0] = indL+ 1;
                        INTEGER(Uopt)[0] = indU + 1;
                        for(i = 0; i < lengthV; i++){
                            REAL(pavaFin)[i] = pavaAux[i];
                        }
                    } 

                //En caso contrario, hay que hacer PAVA decreciente   
                } else {

                    //Si el trozo decreciente no empieza en el primer índice ni termina
                    //en el último
                    if(indU != 0 && indL != lengthV-1){

                        //Cálculo del PAVA decreciente
                        lengthPavaUL = indL - indU - 1;
                        pavaDecreciente(REAL(v), indU+1, indL-1, pavaUL);

                        //Si no se cumple la condición por la izquierda, no se continúa
                        if(pavaUL[lengthPavaUL-1] < REAL(v)[indL]){
                            break;
                        }

                        //Si es válido, se calcula el MSE
                        if(pavaUL[0] <= REAL(v)[indU]){

                            //Ajuste completo
                            j = 0;
                            for(i = lengthV - indL; i < lengthPavaLU; i++){
                                pavaAux[j] = pavaLUrecuperado[i];
                                j++;
                            }
                            for(i = 0; i < lengthPavaUL; i++){
                                pavaAux[j] = pavaUL[i];
                                j++;
                            }
                            for(i = 0; i < lengthV - indL; i++){
                                pavaAux[j] = pavaLUrecuperado[i];
                                j++;                        
                            }
                            mseAux = calculoMSE(lengthV, REAL(v), pavaAux);    

                            if(mseAux < REAL(mseFin)[0]){
                                REAL(mseFin)[0] = mseAux;
                                INTEGER(Lopt)[0] = indL+ 1;
                                INTEGER(Uopt)[0] = indU + 1;
                                for(i = 0; i < lengthV; i++){
                                    REAL(pavaFin)[i] = pavaAux[i];
                                }
                            }
                        }                        

                    //Si el trozo decreciente sí empieza en el primer índice pero
                        //no acaba en el último
                    } else if (indU == 0 && indL != lengthV-1){

                        lengthPavaUL = indL-1;
                        pavaDecreciente(REAL(v), 1, indL-1, pavaUL);

                        //Si no se cumple la condición por la izquierda, no se continúa
                        if(pavaUL[lengthPavaUL-1] < REAL(v)[indL]){
                            break;
                        }

                        //Si es válido, se calcula el MSE
                        if(pavaUL[0] <= REAL(v)[indU]){

                            //Ajuste completo
                            j = 0;
                            pavaAux[j] = pavaLUrecuperado[lengthPavaLU-1];
                            j++;
                            for(i = 0; i < lengthPavaUL; i++){
                                pavaAux[j] = pavaUL[i];
                                j++;
                            }
                            for(i = 0; i < lengthPavaLU-1; i++){
                                pavaAux[j] = pavaLUrecuperado[i];
                                j++;                        
                            }
                            mseAux = calculoMSE(lengthV, REAL(v), pavaAux);    

                            if(mseAux < REAL(mseFin)[0]){
                                REAL(mseFin)[0] = mseAux;
                                INTEGER(Lopt)[0] = indL+ 1;
                                INTEGER(Uopt)[0] = indU + 1;
                                for(i = 0; i < lengthV; i++){
                                    REAL(pavaFin)[i] = pavaAux[i];
                                }
                            }
                        }

                    // El trozo decreciente no empieza en el primer 
                    // índice pero termina en el último
                    } else if (indL == lengthV-1 && indU != 0){

                        lengthPavaUL = indL - indU - 1;
                        pavaDecreciente(REAL(v), indU+1, indL-1, pavaUL);

                        //Si no se cumple la condición por la izquierda, no se continúa
                        if(pavaUL[lengthPavaUL-1] < REAL(v)[indL]){
                            break;
                        }

                        //Si es válido, se calcula el MSE
                        if(pavaUL[0] <= REAL(v)[indU]){

                            //Ajuste completo
                            j = 0;
                            for(i = 1; i < lengthPavaLU; i++){
                                pavaAux[j] = pavaLUrecuperado[i];
                                j++;                        
                            }
                            for(i = 0; i < lengthPavaUL; i++){
                                pavaAux[j] = pavaUL[i];
                                j++;
                            }
                            pavaAux[j] = pavaLUrecuperado[0];

                            mseAux = calculoMSE(lengthV, REAL(v), pavaAux);    

                            if(mseAux < REAL(mseFin)[0]){
                                REAL(mseFin)[0] = mseAux;
                                INTEGER(Lopt)[0] = indL+ 1;
                                INTEGER(Uopt)[0] = indU + 1;
                                for(i = 0; i < lengthV; i++){
                                    REAL(pavaFin)[i] = pavaAux[i];
                                }
                            }
                        }

                    //Última posibilidad: indL = lengthV-1, indU = 0
                    } else {

                        lengthPavaUL = indL - indU - 1;
                        pavaDecreciente(REAL(v),indU+1, indL-1, pavaUL);

                        //Si no se cumple la condición por la izquierda, no se continúa
                        if(pavaUL[lengthPavaUL-1] < REAL(v)[indL]){
                            break;
                        }

                        //Si es válido, se calcula el MSE
                        if(pavaUL[0] <= REAL(v)[indU]){

                            //Ajuste completo
                            j = 0;
                            pavaAux[j] = pavaLUrecuperado[1];
                            j++;
                            for(i = 0; i < lengthPavaUL; i++){
                                pavaAux[j] = pavaUL[i];
                                j++;
                            }
                            pavaAux[j] = pavaLUrecuperado[0];

                            mseAux = calculoMSE(lengthV, REAL(v), pavaAux);    

                            if(mseAux < REAL(mseFin)[0]){
                                REAL(mseFin)[0] = mseAux;
                                INTEGER(Lopt)[0] = indL+ 1;
                                INTEGER(Uopt)[0] = indU + 1;
                                for(i = 0; i < lengthV; i++){
                                    REAL(pavaFin)[i] = pavaAux[i];
                                }
                            }
                        }
                    }
                }
            }   
        }

        //Liberar espacio de los pavas crecientes
        for(i = 0; i < nCandUValidos; i++){
            free(pavasCrecientes[i]);
        }

        if(REAL(mseFin)[0] == 0){
            //No existe un ajuste mejor
            break;
        }
    }

    //Liberar memoria
    free(ordenComprobacion);
    free(v2);
    free(candUValidos);
    free(candUValidosreversed);
    free(pavasCrecientes);
    free(pavaLU);
    free(pavaUL);
    free(lengthPavasCrecientes);
    free(pavaAux);

    //Devolver el resultado
    UNPROTECT(1);

    //Si el mse devuelto es -1, es que no existe solución
    if(REAL(mseFin)[0] >= DBL_MAX - 0.5){
        REAL(mseFin)[0] = -1;
    }
    return(ans);
}



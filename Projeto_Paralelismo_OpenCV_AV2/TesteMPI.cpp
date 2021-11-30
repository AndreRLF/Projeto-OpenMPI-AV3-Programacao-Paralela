#include <stdio.h>
#include <mpi.h> 
#include <omp.h>
#include <opencv2/opencv.hpp>

using namespace cv;

Mat redTones(Mat imagem, int rowStart, int rowEnd) {
    Mat imageR = imagem.clone();
    int tamanho = imageR.size().width;

    int n_procs = omp_get_num_procs();
    omp_set_num_threads(n_procs * 2);
    omp_set_nested(1);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int y = rowStart; y < rowEnd; y++) {
            #pragma omp parallel
            {
                #pragma omp for nowait
                for (int x = 0; x < tamanho; x++) {

                    Vec3b& color = imageR.at<Vec3b>(y, x);

                    color[0] = 0;
                    color[1] = 0;

                    imageR.at<Vec3b>(Point(x, y)) = color;
                }
            }
            
        }
    } 

    omp_set_nested(0);
    return imageR;
}

Mat greenTones(Mat imagem, int rowStart, int rowEnd) {
    Mat imageR = imagem.clone();
    int tamanho = imageR.size().width;

    int n_procs = omp_get_num_procs();
    omp_set_num_threads(n_procs * 2);
    omp_set_nested(1);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int y = rowStart; y < rowEnd; y++) {
            #pragma omp parallel
            {
                #pragma omp for nowait
                for (int x = 0; x < tamanho; x++) {

                    Vec3b& color = imageR.at<Vec3b>(y, x);

                    color[0] = 0;
                    color[2] = 0;

                    imageR.at<Vec3b>(Point(x, y)) = color;
                }
            }
            
        }
    } 

    omp_set_nested(0);
    return imageR;
}

Mat blueTones (Mat imagem, int rowStart, int rowEnd) {
    Mat imageR = imagem.clone();
    int tamanho = imageR.size().width;

    int n_procs = omp_get_num_procs();
    omp_set_num_threads(n_procs * 2);
    omp_set_nested(1);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int y = rowStart; y < rowEnd; y++) {
            #pragma omp parallel
            {
                #pragma omp for nowait
                for (int x = 0; x < tamanho; x++) {

                    Vec3b& color = imageR.at<Vec3b>(y, x);

                    color[1] = 0;
                    color[2] = 0;

                    imageR.at<Vec3b>(Point(x, y)) = color;
                }
            }
            
        }
    } 

    omp_set_nested(0);
    return imageR;
}

Mat grayScale(Mat imagem, int rowStart, int rowEnd) {
    Mat imageR = imagem.clone();
    int tamanho = imageR.size().width;

    int n_procs = omp_get_num_procs();
    omp_set_num_threads(n_procs * 2);
    omp_set_nested(1);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int y = rowStart; y < rowEnd; y++) {
            #pragma omp parallel
            {
                #pragma omp for nowait
                for (int x = 0; x < tamanho; x++) {

                    Vec3b& color = imageR.at<Vec3b>(y, x);

                    int cinza = (color[0] + color[1] + color[2]) / 3;

                    color[0] = cinza;
                    color[1] = cinza;
                    color[2] = cinza;

                    imageR.at<Vec3b>(Point(x, y)) = color;
                }
            }
            
        }
    } 

    omp_set_nested(0);
    return imageR;
}

Mat bw(Mat imagem, int rowStart, int rowEnd) {
    Mat imageR = imagem.clone();
    int tamanho = imageR.size().width;

    int n_procs = omp_get_num_procs();
    omp_set_num_threads(n_procs * 2);
    omp_set_nested(1);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int y = rowStart; y < rowEnd; y++) {
            #pragma omp parallel
            {
                #pragma omp for nowait
                for (int x = 0; x < tamanho; x++) {

                    Vec3b& color = imageR.at<Vec3b>(y, x);

                    int cor = 0;
                    int cinza = (color[0] + color[1] + color[2]) / 3;

                    if (cinza > 128)
                        cor = 255;

                    color[0] = cor;
                    color[1] = cor;
                    color[2] = cor;

                    imageR.at<Vec3b>(Point(x, y)) = color;
                }
            }
            
        }
    } 

    omp_set_nested(0);
    return imageR;
}

int main(int argc, char* argv[]) {

    int rank, size;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        Mat img = imread("images/pesada2.jpg", IMREAD_COLOR);
        Mat processedImg = img.clone();
        int largura = img.size().width;
        int altura = img.size().height;        

        namedWindow("Imagem Original", WINDOW_NORMAL);
        imshow("Imagem Original", img);
        waitKey();

        int imgLines = img.size().height;
        int part = imgLines / (size - 1);

        int inicio = 0;
        int fim = part;

        for (int i = 1; i < size; i++) {
            if (i == (size - 1))
                fim = imgLines;

            MPI_Send(&inicio, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&fim, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&altura, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
            MPI_Send(&largura, 1, MPI_INT, i, 3, MPI_COMM_WORLD);
            MPI_Send(processedImg.data, largura * altura * 3, MPI_UNSIGNED_CHAR, i, 4, MPI_COMM_WORLD);

            MPI_Recv(processedImg.data, largura * altura * 3, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, &status);

            inicio = fim;
            fim += part;
        }

        namedWindow("Imagem Cinza", WINDOW_NORMAL);
        imshow("Imagem Cinza", processedImg);
        waitKey();

        bool result = imwrite("saved/saved_img.png", processedImg);
        if (result)
            std::cout << "\nImagem Salva com Sucesso!\n";
        else
            std::cout << "\nERRO ao salvar imagem!\n";
    }
    else {
        int initAux, endAux, alturaAux, larguraAux;
        MPI_Recv(&initAux, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&endAux, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&alturaAux, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
        MPI_Recv(&larguraAux, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);

        Mat imgAux = Mat(Size(larguraAux, alturaAux), CV_8UC3);
        MPI_Recv(imgAux.data, larguraAux * alturaAux * 3, MPI_UNSIGNED_CHAR, 0, 4, MPI_COMM_WORLD, &status);

        //Processamento da Imagem
        imgAux = bw(imgAux, initAux, endAux);
        MPI_Send(imgAux.data, larguraAux * alturaAux * 3, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
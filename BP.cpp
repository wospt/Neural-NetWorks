// BP.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include<stdlib.h>
#include<math.h>
#include<stdio.h>
#include<time.h>

#define COUNT 6
#define IN_NUM 3		//��������
#define OUT_NUM 2		//��������
#define numHidden 30	//���ز����

double weight_hidden[IN_NUM][numHidden];		//����㵽���ز��Ȩ��
double bias_hidden[numHidden];					//ƫ����
double weight_output[numHidden][OUT_NUM];		//���ز㵽������Ȩ��
double bias_output[OUT_NUM];					
double learnRate = 0.8;							//ѧϰ����
double accuracy = 0.000001;						//���ȿ���
int maxLoopCount = 1000000;						//���ѭ������

double fnet(double net)							//sigmod����
{
	return 1 / (1 + exp(-net));
}
double dfnet(double net)						//sigmod������
{
	return net * (1 - net);
}

//BP�������ʼ��
int InitBp()
{
	int i, j;
	srand((unsigned)time(NULL));
	for (i = 0; i < IN_NUM; i++)
		for (j = 0; j < numHidden; j++)
		{
			weight_hidden[i][j] = rand() / (double)(RAND_MAX)-0.5;
			bias_hidden[j] = rand() / (double)(RAND_MAX)-0.5;
		}
	for (i = 0; i < numHidden; i++)
		for (j = 0; j < OUT_NUM; j++)
		{
			weight_output[i][j] = rand() / (double)(RAND_MAX)-0.5;
			bias_output[j] = rand() / (double)(RAND_MAX)-0.5;
		}
	return 1;
}

//ѵ��BP����
int TrainBp(float x[COUNT][IN_NUM], float y[COUNT][OUT_NUM])
{
	double delta_hidden[numHidden], delta_output[OUT_NUM];
	double output_hidden[numHidden], output_output[OUT_NUM];
	int i, j, k, n;
	double temp;

	double loss = accuracy + 1;   //Ŀ����Ϊ����loss����f���ܽ��������ѭ��  
	for (n = 0; loss > accuracy && n < maxLoopCount; n++)
	{
		loss = 0;
		for (i = 0; i < COUNT; i++)
		{
			//ǰ�����  
			for (k = 0; k < numHidden; k++)
			{
				temp = 0;
				for (j = 0; j < IN_NUM; j++)
					temp += x[i][j] * weight_hidden[j][k];
				output_hidden[k] = fnet(temp + bias_hidden[k]);
			}
			for (k = 0; k < OUT_NUM; k++)
			{
				temp = 0;
				for (j = 0; j < numHidden; j++)
					temp += output_hidden[j] * weight_output[j][k];
				output_output[k] = fnet(temp + bias_output[k]);
			}

			//�������  
			for (j = 0; j < OUT_NUM; j++)
				loss += 0.5 * (y[i][j] - output_output[j]) * (y[i][j] - output_output[j]);

			//�������  
			for (j = 0; j < OUT_NUM; j++)
				delta_output[j] = (y[i][j] - output_output[j]) * dfnet(output_output[j]);

			for (j = 0; j < numHidden; j++)
				for (k = 0; k < OUT_NUM; k++)
					weight_output[j][k] += learnRate * delta_output[k] * output_hidden[j];
			for (k = 0; k < OUT_NUM; k++)
				bias_output[k] += learnRate * delta_output[k];


			for (j = 0; j < numHidden; j++)
			{
				temp = 0;
				for (k = 0; k < OUT_NUM; k++)
					temp += weight_output[j][k] * delta_output[k];
				delta_hidden[j] = temp * dfnet(output_hidden[j]);
			}

			for (j = 0; j < IN_NUM; j++)
				for (k = 0; k < numHidden; k++)
					weight_hidden[j][k] += learnRate * delta_hidden[k] * x[i][j]; //���ز�Ȩ����  
			for (k = 0; k < numHidden; k++)
				bias_hidden[k] += learnRate * delta_hidden[k];
		}
		if (n % 10 == 0)
			printf("��� : %f\n", loss);          //ÿѵ��10�����һ�������  
	}
	printf("�ܹ�ѭ��������%d\n", n);

	printf("bp����ѵ��������\n");

	return 1;
}

int UseBp()
{
	float Input[IN_NUM];
	double output_hidden[numHidden];			//���ز����
	double output_output[OUT_NUM];				//��������
	while (1)		//����ִ�У������жϳ���
	{
		printf("������3������ \n");
		int i, j;
		for (i = 0; i < IN_NUM; i++)
			scanf("%f", &Input[i]);

		double temp;
		for (i = 0; i < numHidden; i++)
		{
			temp = 0;
			for (j = 0; j < IN_NUM; j++)
				temp += Input[j] * weight_hidden[j][i];
			output_hidden[i] = fnet(temp + bias_hidden[i]);
		}
		for (i = 0; i < OUT_NUM; i++)
		{
			temp = 0;
			for (j = 0; j < numHidden; j++)
				temp += output_hidden[j] * weight_hidden[j][i];
			output_output[i] = fnet(temp + bias_output[i]);
		}

		printf("�����  ");
		for (i = 0; i < OUT_NUM; i++)
			printf("%.3f", output_output[i]);
		printf("\n");
	}
	return 1;
}
int _tmain(int argc, _TCHAR* argv[])
{
	//x---����������y---�������  
	//���룺0.8, 0.5,   0---------���0,1  
	//���룺0.9, 0.7, 0.3---------���0,1  
	//���룺  1, 0.8, 0.5---------���0,1  
	//���룺  0, 0.2, 0.3---------���1,0                      
	//���룺0.2, 0.1, 1.3---------���1,0                              
	//���룺0.2, 0.7, 0.8---------���1,0      
	float x[COUNT][IN_NUM] = { { 0.8, 0.5, 0 },
	{ 0.9, 0.7, 0.3 },
	{ 1, 0.8, 0.5 },
	{ 0, 0.2, 0.3 },
	{ 0.2, 0.1, 1.3 },
	{ 0.2, 0.7, 0.8 } }; //ѵ������  
	float y[COUNT][OUT_NUM] = { { 0, 1 },
	{ 0, 1 },
	{ 0, 1 },
	{ 1, 0 },
	{ 1, 0 },
	{ 1, 0 } };      //�������  

	InitBp();                    //��ʼ��bp����ṹ  
	TrainBp(x, y);             //ѵ��bp������  
	UseBp();                     //����bp������  
	return 1;
}


// nvcc horizontalDustDisk.cu -o dust -lglut -lm -lGLU -lGL
//To force kill hit "control c" in the window you launched it from.
// #include <gtk/gtk.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>
using namespace std;

#define PI 3.141592654
#define BLOCK 256

#define TOP_VIEW 1
#define SIDE_VIEW 2
#define FRUSTRUM 3
#define ORTHO 4
#define BOTTOM_RING 20
#define BASE_ION_UP 21
#define BASE_ION_DOWN 22

struct ionWakeInfoStructure
{
	int companionId;
	float dy;    
	float d; 
};

FILE* MovieFile;
int* Buffer;
int MovieOn;

// Globals to be read in from parameter file.
int		NumberOfDustParticles;

double 	Gravity;
double 	DustDensity;
double 	BaseDustDiameter;
double	DustDiameterStandardDeviation;

double 	CoulombConstant;
double	ElectronCharge;
double 	BaseElectronsPerUnitDiameter;
double	electronStdPerUnitDiameter;
double	DebyeLength;
double	CutOffMultiplier;
double	SheathHeight;

double 	BaseIonWakeChargePercent;
double 	BaseIonWakeLength;

double 	CavityCharge;
double 	RadiusOfCavity;
double 	HieghtOfCavity;
double 	BottomePlatesCharge;

double 	Drag;

double 	Dt;
int 	DrawRate;
int 	PrintTimeRate;

// Globals to hold our unit convertions.
double MassUnit;
double LengthUnit;
double ChargeUnit;
double TimeUnit;

// Call back globals
int Pause;
int LaserOn;
int Trace;
int MouseOn;
int TranslateRotate = 1;
float4 CenterOfView;
float4 AngleOfView;
int DrawBottomRing;
int SelectedDustGrainId1, SelectedDustGrainId2, SelectedDustGrainId3;

// Timing globals
int DrawTimer;
int PrintTimer;
float RunTime;

// Position, velocity, force and ionwake globals
float4 *DustPositionCPU, *DustVelocityCPU, *DustForceCPU, *IonWakeCPU;
float4 *DustPositionGPU, *DustVelocityGPU, *DustForceGPU, *IonWakeGPU;
float4 *DustColor;
int *IonWakeNeighborGPU;
ionWakeInfoStructure *IonWakeInfoCPU;
ionWakeInfoStructure *IonWakeInfoGPU;

// CUDA globals
dim3 Block, Grid;

// Window globals
static int Window;
int XWindowSize;
int YWindowSize; 
double Near;
double Far;
double EyeX;
double EyeY;
double EyeZ;
double CenterX;
double CenterY;
double CenterZ;
double UpX;
double UpY;
double UpZ;

// Prototyping functions
void readSimulationParameters();
void setUnitConvertions();
void PutConstantsIntoOurUnits();
void allocateMemory();
void setInitialConditions(); 
void drawPicture();
void n_body();
void errorCheck(const char*);
void setup();

#include "./callBackFunctions.h"

void readSimulationParameters()
{
	ifstream data;
	string name;
	
	data.open("./simulationSetup");
	
	if(data.is_open() == 1)
	{
		getline(data,name,'=');
		data >> NumberOfDustParticles;
		
		getline(data,name,'=');
		data >> Gravity;
		
		getline(data,name,'=');
		data >> DustDensity;
		
		getline(data,name,'=');
		data >> BaseDustDiameter;
		
		getline(data,name,'=');
		data >> DustDiameterStandardDeviation;
		
		getline(data,name,'=');
		data >> CoulombConstant;
		
		getline(data,name,'=');
		data >> ElectronCharge;
		
		getline(data,name,'=');
		data >> BaseElectronsPerUnitDiameter;
		
		getline(data,name,'=');
		data >> electronStdPerUnitDiameter;
		
		getline(data,name,'=');
		data >> DebyeLength;
		
		getline(data,name,'=');
		data >> CutOffMultiplier;
		
		getline(data,name,'=');
		data >> SheathHeight;
		
		getline(data,name,'=');
		data >> BaseIonWakeChargePercent;
		
		getline(data,name,'=');
		data >> BaseIonWakeLength;
		
		getline(data,name,'=');
		data >> CavityCharge;
		
		getline(data,name,'=');
		data >> RadiusOfCavity;
		
		getline(data,name,'=');
		data >> HieghtOfCavity;
		
		getline(data,name,'=');
		data >> BottomePlatesCharge;
		
		getline(data,name,'=');
		data >> Drag;
		
		getline(data,name,'=');
		data >> Dt;
		
		getline(data,name,'=');
		data >> DrawRate;
		
		getline(data,name,'=');
		data >> PrintTimeRate;
	}
	else
	{
		printf("\nTSU Error could not open simulationSetup file\n");
		exit(0);
	}
	
	printf("\n These are what all the basic constants that were read in for a spot check\n");
	printf("\n NumberOfDustParticles = %d", NumberOfDustParticles);
	printf("\n Gravity = %f meters per second squared", Gravity);
	printf("\n DustDensity = %f grams per centimeter cubed ", DustDensity);
	printf("\n BaseDustDiameter = %f microns", BaseDustDiameter);
	printf("\n DustDiameterStandardDeviation = %f microns", DustDiameterStandardDeviation);
	printf("\n CoulombConstant = %e grams meters cubed second to the minus 2 coulomb to the minus 2", CoulombConstant);
	printf("\n ElectronCharge = %e coulombs", ElectronCharge);
	printf("\n BaseElectronsPerUnitDiameter = %f electrons per micron", BaseElectronsPerUnitDiameter);
	printf("\n BaseElectronsPerUnitDiameter*BaseDustDiameter = %f number of electrons on a standard dust grain.", BaseElectronsPerUnitDiameter*BaseDustDiameter);
	printf("\n electronStdPerUnitDiameter = %f electron fluxuation per micron", electronStdPerUnitDiameter);
	printf("\n electronStdPerUnitDiameter*BaseDiameter = %f electron fluxuation on a stadard dust grain", electronStdPerUnitDiameter*BaseDustDiameter);
	printf("\n DebyeLength = %f microns", DebyeLength);
	printf("\n CutOffMultiplier = %f", CutOffMultiplier);
	printf("\n SheathHeight = %f microns", SheathHeight);
	printf("\n BaseIonWakeChargePercent = %f percent of dust charge", BaseIonWakeChargePercent);
	printf("\n BaseIonWakeLength = %f microns", BaseIonWakeLength);
	printf("\n CavityCharge = %e ???", CavityCharge);
	printf("\n RadiusOfCavity = %f centimeters", RadiusOfCavity);
	printf("\n HieghtOfCavity = %f centimeters", HieghtOfCavity);
	printf("\n BottomePlatesCharge = %e kilograms*second-2*coulomb-1", BottomePlatesCharge);
	printf("\n Drag = %e ???", Drag);
	printf("\n Dt = %f number of divisions of the final time unit.", Dt);
	printf("\n DrawRate = %d Dts between picture draws", DrawRate);
	printf("\n PrintTimeRate = %d Dts between time prints to the screen", PrintTimeRate);
	
	data.close();
	printf("\n\n ********************************************************************************");
	printf("\n Parameter file has been read");
	printf("\n ********************************************************************************\n");
}

void allocateMemory()
{
	Block.x = BLOCK;
	Block.y = 1;
	Block.z = 1;
	
	Grid.x = (NumberOfDustParticles - 1)/Block.x + 1;
	Grid.y = 1;
	Grid.z = 1;
	
	DustPositionCPU = (float4*)malloc(NumberOfDustParticles*sizeof(float4));
	DustVelocityCPU = (float4*)malloc(NumberOfDustParticles*sizeof(float4));
	DustForceCPU    = (float4*)malloc(NumberOfDustParticles*sizeof(float4));
	IonWakeCPU    	= (float4*)malloc(NumberOfDustParticles*sizeof(float4));
	DustColor    	= (float4*)malloc(NumberOfDustParticles*sizeof(float4));
	IonWakeInfoCPU	= (ionWakeInfoStructure*)malloc(NumberOfDustParticles*sizeof(ionWakeInfoStructure));
	
	cudaMalloc( (void**)&DustPositionGPU, NumberOfDustParticles*sizeof(float4));
	errorCheck("cudaMalloc DustPositionGPU");
	cudaMalloc( (void**)&DustVelocityGPU, NumberOfDustParticles*sizeof(float4));
	errorCheck("cudaMalloc DustVelocityGPU");
	cudaMalloc( (void**)&DustForceGPU,    NumberOfDustParticles*sizeof(float4));
	errorCheck("cudaMalloc DustForceGPU");
	cudaMalloc( (void**)&IonWakeGPU,    NumberOfDustParticles*sizeof(float4));
	errorCheck("cudaMalloc IonWakeGPU");
	cudaMalloc( (void**)&IonWakeInfoGPU, NumberOfDustParticles*sizeof(ionWakeInfoStructure));
	errorCheck("cudaMalloc IonWakeInfoGPU");
	
	printf("\n\n ********************************************************************************");
	printf("\n Memory has been allocated");
	printf("\n ********************************************************************************\n");
}

void setUnitConvertions()
{	
	/*
	Mass:
	Because the only mass we will be dealing with is a dust particle lets let the mass of a dust particle be 1. 
	Now how many grams is that? 
	The density of a dust particle is DensityOfDustParticle g/cm^3. 
	The diameter of a dust particle is DiameterOfDustParticle microns.
	If we assume the shape of a dust particlecto be a sphere the volume will be 4/3*PI*r^3.
	Lets get micometers to cm so the radius of a dust particle is (DiameterOfDustParticle/2.0)E-4 cm.
	Hence the mass of a dust particle is DensityOfDustParticle*g*(cm^-3)*(4.0/3.0)*PI*((DiameterOfDustParticle/2.0)E-12)*(cm^3).
	So one of our mass units is DensityOfDustParticle*(4.0/3.0)*PI*((DiameterOfDustParticle/2.0)E-12) grams.
	*/
	MassUnit = (DustDensity*(4.0/3.0)*PI*(BaseDustDiameter/(2.0*10000.0))*(BaseDustDiameter/(2.0*10000.0))*(BaseDustDiameter/(2.0*10000.0)));
	
	/*
	Length:
	Because the most important distance is the distance between two adjacent dust particles at equalibrium lets let that be our distance unit.
	To get started let assume we are working with a RadiusOfCavity cm radius well and a NumberOfDustParticles dust particles.
	For simplicity lets put an evenly spaced square latice in a RadiusOfCavity circle. 
	The sides would be 2*RadiusOfCavity/(2^(1/2)) cm long.
	The number of dust particle in each row would be NumberOfDustParticles^(1/2).
	The number of spacing in each row would be NumberOfDustParticles^(1/2) - 1.
	Hense the length of each spacing would be RadiusOfCavity*(2^(1/2))/(NumberOfDustParticles^(1/2) - 1) centimeters.
	Lets put this in meters.
	So our length unit is RadiusOfCavity*(2^(1/2))/(NumberOfDustParticles^(1/2) - 1)/100 meters.
	*/
	LengthUnit = ((2.0*RadiusOfCavity/sqrt(2.0))/(sqrt(NumberOfDustParticles) - 1.0))/100.0;
	
	/*
	Charge:
	Because the most important charge is the charge on a dust particle this should be around our charge unit.
	If we assume n electrons per micro of dust particle diameter we have our charge unit as 
	DiameterOfDustParticle(in microns)*n*1.60217662E-19 coulombs.
	So our charge unit is n*DiameterOfDustParticle*1.602E-19 C.
	*/
	//ChargeUnit = DiameterOfDustParticle*1.60217662e-16;
	ChargeUnit = BaseElectronsPerUnitDiameter*BaseDustDiameter*ElectronCharge;
	
	/*
	Time:
	Let's set time so that the Coulomb constant is 1. That just makes calulations easier.
	Assuming K at 8.9875517923E9 kg*m^3*s^-2*C^-2 or 8.9875517923E12 g*m^3*s^-2*C^-2*
	If I did every thing right I get our time unit should be the square root of massUnit*LengthUnit^3*ChargeUnit^-2*(8.9875517923E12)^-1 seconds.
	*/
	TimeUnit = sqrt(MassUnit*LengthUnit*LengthUnit*LengthUnit/(ChargeUnit*ChargeUnit*CoulombConstant));
	
	// To change a mass, length, charge or time from the run units into grams, meters, coulombs or seconds just multiple by the apropriate unit from here.
	// To change a mass, length, charge or time from grams, meters, coulombs or seconds into our units just divide by the apropriate unit from here.
	printf("\n This is what all of the convertions are for a spot check\n");
	printf("\n Our MassUnit 	= %e grams", MassUnit);
	printf("\n Our LengthUnit = %e meters", LengthUnit);
	printf("\n Our ChargeUnit = %e coulombs", ChargeUnit);
	printf("\n Our TimeUnit 	= %e seconds", TimeUnit);
	
	printf("\n\n ********************************************************************************");
	printf("\n Unit convertions have been set.");
	printf("\n ********************************************************************************\n");
}

void PutConstantsIntoOurUnits()
{
	// Putting all the parameters into our units.
	
	// NumberOfDustParticles just a number no need to convert.
	
	// Gravity is in meters per secind square so in of units we need to multiplu by TimeUnit^2*LengthUnit^-1.
	Gravity *= TimeUnit*TimeUnit/LengthUnit;
	
	// Dust density is in grams per cenimeter cubed so need to take this to meters then divive by (MassUnit/LengthUnit^2);
	DustDensity *= 100.0*100.0*100.0;
	DustDensity /= (MassUnit/(LengthUnit*LengthUnit*LengthUnit));
	
	// BaseDustDiameter is in microns so take it to meters then divide by lengthUnit.
	BaseDustDiameter /= 1.0e6;
	BaseDustDiameter /= LengthUnit;
	
	// Standard deviation is in microns so take it to meters then divide by lengthUnit.
	DustDiameterStandardDeviation /= 1.0e6;
	DustDiameterStandardDeviation /= LengthUnit;
	
	// The coulomb constant should be 1 because of the way we set the time unit (see TimeUnit above).
	CoulombConstant = 1.0; 
	
	// Putting the charge of an electron into our units.
	ElectronCharge /= ChargeUnit;
	
	// Putting the number of electrons per unit of diameter into our units. They were in electrons per micron so take them to meters then to our units.
	BaseElectronsPerUnitDiameter *= 1.0e6;
	BaseElectronsPerUnitDiameter *= LengthUnit;
	
	// Putting the number of electrons per unit of diameter into our units. They were in electrons per micron so take them to meters then to our units.
	electronStdPerUnitDiameter *= 1.0e6;
	electronStdPerUnitDiameter *= LengthUnit;
	
	// Puting the debye length into our units. It is in micros so take it to meters then to our units.
	DebyeLength /= 1.0e6;
	DebyeLength /= LengthUnit;
	
	// Puting the sheath height into our units. It is in millimeters so take it to meters then to our units.
	SheathHeight /= 1.0e3;
	SheathHeight /= LengthUnit;
	
	// BaseIonWakeChargePercent No change needed here because this is a percent of the dust charge it is attached to.
	
	// This is in microns so take it to meters then to our units.
	BaseIonWakeLength /= 1.0e6;
	BaseIonWakeLength /= LengthUnit;
	
	// This is a force I multiple by the dust charge to push it (electrically) away from the cavity wall. It is given in grams*seconds^-2*coulomb^-1.
	// To get it into our unit you need to divide by the apropriate units.
	CavityCharge = CavityCharge*TimeUnit*TimeUnit*ChargeUnit/MassUnit;
	
	// Taking cavity dimitions in cm to our units. First take them to meters then to our units.
	RadiusOfCavity /= 100.0;
	RadiusOfCavity /= LengthUnit;
	HieghtOfCavity /= 100.0;
	HieghtOfCavity /= LengthUnit;
	
	// This is the force that you multiply times the charge and distance from the bottom plate to get a force.
	// It is given in kilograms*seconds^-2*coulomb^-1.
	// To get it into our unit you need to divide by the apropriate units.
	// First take kilograms to grams.
	BottomePlatesCharge = BottomePlatesCharge*1.0e3;
	BottomePlatesCharge = BottomePlatesCharge*TimeUnit*TimeUnit*ChargeUnit/MassUnit;
	
	// This needs to be changed into our units ??????????????
	Drag *= 1.0;
	
	// CutOffMultiplier is just a multiplier so no adjustment is needed.
	// Dt is a percent of the time unit so no need to change
	// DrawRate is just a number of step between drawing so no change needed.
	// PrintTimeRate is just a number of step between Printing the time to the screen so no change needed.
	
	printf("\n These are what all the basic constants in our units for a spot check\n");
	printf("\n NumberOfDustParticles = %d", NumberOfDustParticles);
	printf("\n Gravity = %e", Gravity);
	printf("\n DustDensity = %e", DustDensity);
	printf("\n BaseDustDiameter = %e", BaseDustDiameter);
	printf("\n DustDiameterStandardDeviation = %e", DustDiameterStandardDeviation);
	printf("\n CoulombConstant = %e", CoulombConstant);
	printf("\n ElectronCharge = %e", ElectronCharge);
	printf("\n BaseElectronsPerUnitDiameter = %e", BaseElectronsPerUnitDiameter);
	printf("\n BaseElectronsPerUnitDiameter*BaseDustDiameter = %e", BaseElectronsPerUnitDiameter*BaseDustDiameter);
	printf("\n electronStdPerUnitDiameter = %e", electronStdPerUnitDiameter);
	printf("\n electronStdPerUnitDiameter*BaseDiameter = %e", electronStdPerUnitDiameter*BaseDustDiameter);
	printf("\n DebyeLength = %e", DebyeLength);
	printf("\n CutOffMultiplier = %e", CutOffMultiplier);
	printf("\n SheathHeight = %e", SheathHeight);
	printf("\n BaseIonWakeChargePercent = %e", BaseIonWakeChargePercent);
	printf("\n BaseIonWakeLength = %e", BaseIonWakeLength);
	printf("\n CavityCharge = %e", CavityCharge);
	printf("\n RadiusOfCavity = %e", RadiusOfCavity);
	printf("\n HieghtOfCavity = %e", HieghtOfCavity);
	printf("\n BottomePlatesCharge = %e", BottomePlatesCharge);
	printf("\n Drag = %e", Drag);
	printf("\n Dt = %e", Dt);
	printf("\n DrawRate = %d", DrawRate);
	printf("\n PrintTimeRate = %d", PrintTimeRate);
	
	printf("\n\n ********************************************************************************");
	printf("\n Constants have been put into our units.");
	printf("\n ********************************************************************************\n");
}

void setInitialConditions()
{
	int test;
	double temp1, temp2;
	double mag, radius, seperation;
	double diameter, mass, charge, randomNumber; // numberOfElectronsPerUnitDiameter;
	int numberOfElectrons;
	time_t t;
	
	// Seading the random number generater.
	srand((unsigned) time(&t));
	
	// Zeroing everything out just to be safe and setting the base collor. 
	for(int i = 0; i < NumberOfDustParticles; i++)
	{
		DustPositionCPU[i].x = 0.0;
		DustPositionCPU[i].y = 0.0;
		DustPositionCPU[i].z = 0.0;
		DustPositionCPU[i].w = 0.0;
		
		DustVelocityCPU[i].x = 0.0;
		DustVelocityCPU[i].y = 0.0;
		DustVelocityCPU[i].z = 0.0;
		DustVelocityCPU[i].w = 0.0;
		
		DustForceCPU[i].x = 0.0;
		DustForceCPU[i].y = 0.0;
		DustForceCPU[i].z = 0.0;
		DustForceCPU[i].w = 0.0;
		
		IonWakeCPU[i].x = 0.0;
		IonWakeCPU[i].y = 0.0;
		IonWakeCPU[i].z = 0.0;
		IonWakeCPU[i].w = 0.0;
		
		IonWakeInfoCPU[i].companionId = 0;
		IonWakeInfoCPU[i].d = 0.0;
		IonWakeInfoCPU[i].dy = 0.0;
		
		// A nice brown is 0.707 0.395 0.113
		DustColor[i].x = 0.707;
		DustColor[i].y = 0.395;
		DustColor[i].z = 0.113;
		DustColor[i].w = 0.0;
	}

	// The w component of position holds the charge of the dust particle. Because charge is used most in finding the force on a dust grain.
	// The w component of velocity holds the diameter of the dust particle.
	// The w component of the force holds the mass on the dust particle.
	// Note: I do not do this for dust diameter because the length unit is set on dust seperation not dust diameter.
	
	// Setting the dust diameter, which then sets the dust charge and mass
	for(int i = 0; i < NumberOfDustParticles; i++)
	{
		// Getting a log-normal random diameter with mean BaseDustDiameter and standard deviation DustDiameterStandardDeviation.
		// ??? look at this more closely. I'm not sure if I did the log normal correct.
		test = 0;
		while(test ==0)
		{	
			// Getting two uniform random numbers in [0,1]
			temp1 = ((double) rand() / (RAND_MAX));
			temp2 = ((double) rand() / (RAND_MAX));
			
			// Getting ride of the end points so now random number is in (0,1)
			if(temp1 == 0 || temp1 == 1 || temp2 == 0 || temp2 == 1) 
			{
				test = 0;
			}
			else
			{
				// Using Box-Muller to get a standard normal random number.
				randomNumber = cos(2.0*PI*temp2)*sqrt(-2 * log(temp1));
				// Creating a log-normal distrobution from the normal randon number.
				randomNumber = exp(randomNumber);
				test = 1;
			}
		}
		diameter = BaseDustDiameter + DustDiameterStandardDeviation*randomNumber;
		//printf("diameter = %f compute units or %f microns\n", diameter , diameter*LengthUnit*1000000.0);
		radius = diameter/2.0;
		
		// Now the mass of this dust particle will be determined off of its diameter and its density (at present all the dust particles have the same density).
		mass = DustDensity*(4.0/3.0)*PI*radius*radius*radius;
		
		// Setting the charge on the dust grain.
		numberOfElectrons = round(BaseElectronsPerUnitDiameter*diameter);
		charge = numberOfElectrons*ElectronCharge;
		
		DustPositionCPU[i].w = charge;
		DustVelocityCPU[i].w = diameter;
		DustForceCPU[i].w = mass;
	}
	
	// ????? I'm not happy with the mean and std they seem to be a little off.
	// The std seems to be double. This may be from the log normal.
	// Just printing out here for debuging.
	double mean, std;
	mean = 0.0;
	for(int i = 0; i < NumberOfDustParticles; i++)
	{
		mean += DustVelocityCPU[i].w*LengthUnit*1000000.0;
	}
	mean /= NumberOfDustParticles;
	
	std = 0.0;
	for(int i = 0; i < NumberOfDustParticles; i++)
	{
		std += (DustVelocityCPU[i].w*LengthUnit*1000000.0 - mean)*(DustVelocityCPU[i].w*LengthUnit*1000000.0 - mean);
	}
	std /= NumberOfDustParticles;
	std = sqrt(std);
	printf("\n Dust Mean = %f in microns, Standard Deviation = %f", mean , std);
	

	// Setting the initial positions.
	for(int i = 0; i < NumberOfDustParticles; i++)
	{
		test = 0;
		while(test == 0)
		{
			// Get random number between -1 at 1.
			DustPositionCPU[i].x = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			DustPositionCPU[i].z = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			// Setting it to have random radius from 0 to radius of the cavity.
			mag = sqrt(DustPositionCPU[i].x*DustPositionCPU[i].x + DustPositionCPU[i].z*DustPositionCPU[i].z);
			radius = ((float)rand()/(float)RAND_MAX)*RadiusOfCavity;
			if(0.0 < mag)
			{
				DustPositionCPU[i].x *= radius/mag;
				DustPositionCPU[i].z *= radius/mag;
			}
			else
			{
				DustPositionCPU[i].x = 0.0;
				DustPositionCPU[i].z = 0.0;
			}
			DustPositionCPU[i].y = ((float)rand()/(float)RAND_MAX)*HieghtOfCavity/10.0 + HieghtOfCavity - HieghtOfCavity/10.0;
			test = 1;
			
			for(int j = 0; j < i; j++)
			{
				seperation = sqrt((DustPositionCPU[i].x-DustPositionCPU[j].x)*(DustPositionCPU[i].x-DustPositionCPU[j].x) + (DustPositionCPU[i].y-DustPositionCPU[j].y)*(DustPositionCPU[i].y-DustPositionCPU[j].y) + (DustPositionCPU[i].z-DustPositionCPU[j].z)*(DustPositionCPU[i].z-DustPositionCPU[j].z));
				if(seperation < 1.5*DebyeLength) //LengthUnit)
				{
					test = 0;
					break;
				}
			}
		}
	}

	// Setting the initial ionwake. We will use the .x to hold the percent charge being used at the current time step.
	// .z will hold the updated percent charge that will be used for the next time step. They will be swaped out.
	// This will allow us to sync between blocks and update in the move kernal.
	for(int i = 0; i < NumberOfDustParticles; i++)
	{
		IonWakeCPU[i].x = BaseIonWakeChargePercent;
		IonWakeCPU[i].y = BaseIonWakeLength;
		IonWakeCPU[i].z = 0.0;
		IonWakeCPU[i].w = 0.0;
		
		IonWakeInfoCPU[i].companionId = -1;
		IonWakeInfoCPU[i].dy = 100000.0;
		IonWakeInfoCPU[i].d = 100000.0;
	}
	
	printf("\n\n ********************************************************************************");
	printf("\n Initial conditions have been set");
	printf("\n ********************************************************************************\n");
}

void drawPicture()
{
	if(Trace == 0)
	{
		glClear(GL_COLOR_BUFFER_BIT);
		glClear(GL_DEPTH_BUFFER_BIT);
	}
	
	float dustSize = 0.1;
	// Drawing all the dust grains and IonWakes.
	for(int i = 0; i < NumberOfDustParticles; i++)
	{
		// Dust
		glColor3d(DustColor[i].x,DustColor[i].y,DustColor[i].z);
		glPushMatrix();
			glTranslatef(DustPositionCPU[i].x, DustPositionCPU[i].y, DustPositionCPU[i].z);
			glutSolidSphere(dustSize,5,5);
		glPopMatrix();	
		
		// IonWake
		glColor3d(1.0,0.0,0.0);
		glPushMatrix();
			glTranslatef(DustPositionCPU[i].x, DustPositionCPU[i].y - IonWakeCPU[i].y, DustPositionCPU[i].z);
			glutSolidSphere(dustSize*IonWakeCPU[i].x,20,20);
		glPopMatrix();	
	}
	
	glLineWidth(1.0);
	float divitions = 60.0;
	float angle = 2.0*PI/divitions;
	
	// Drawing the top of cavity ring.
	for(int i = 0; i < divitions; i++)
	{
		if(i < divitions/2) glColor3d(1.0,0.0,0.0);
		else glColor3d(0.0,0.0,1.0);
		glBegin(GL_LINES);
			glVertex3f(sin(angle*i)*RadiusOfCavity, HieghtOfCavity, cos(angle*i)*RadiusOfCavity);
			glVertex3f(sin(angle*(i+1))*RadiusOfCavity, HieghtOfCavity, cos(angle*(i+1))*RadiusOfCavity);
		glEnd();
	}
	
	// Drawing top of sheath ring.
	glColor3d(0.0,1.0,0.0);
	for(int i = 0; i < divitions; i++)
	{
		glBegin(GL_LINES);
			glVertex3f(sin(angle*i)*RadiusOfCavity, SheathHeight, cos(angle*i)*RadiusOfCavity);
			glVertex3f(sin(angle*(i+1))*RadiusOfCavity, SheathHeight, cos(angle*(i+1))*RadiusOfCavity);
		glEnd();
	}
	
	// Drawing the bottom plate and a cutoff length.
	if(DrawBottomRing == 1)
	{
		glColor3d(1.0,1.0,1.0);
		for(int i = 0; i < divitions; i++)
		{
			glBegin(GL_LINES);
				glVertex3f(sin(angle*i)*RadiusOfCavity, 0.0, cos(angle*i)*RadiusOfCavity);
				glVertex3f(sin(angle*(i+1))*RadiusOfCavity, 0.0, cos(angle*(i+1))*RadiusOfCavity);
			glEnd();
		}
		
		// Drawing a debye length so we can get a prospective.
		glColor3d(0.0,1.0,0.0);
		glBegin(GL_LINES);
				glVertex3f(DustPositionCPU[0].x, DustPositionCPU[0].y, DustPositionCPU[0].z);
				glVertex3f(DustPositionCPU[0].x + CutOffMultiplier*DebyeLength, DustPositionCPU[0].y, DustPositionCPU[0].z);
		glEnd();
	}
	glutSwapBuffers();
	
	// Making a video of the run.
	if(MovieOn == 1)
	{
		glReadPixels(5, 5, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, Buffer);
		fwrite(Buffer, sizeof(int)*XWindowSize*YWindowSize, 1, MovieFile);
	}
}

__global__ void getForces(float4 *dustPos, float4 *dustVel, float4 *dustForce, float4 *ionWake, ionWakeInfoStructure *ionWakeInfoGPU, float baseDustDiameter, float coulombConstant, float debyeLength, float cutOffMultiplier, float radiusOfCavity, float cavityCharge, float hieghtOfCavity, float sheathHeight, float baseIonWakeChargePercent, float bottomePlatesCharge, float gravity, int numberOfParticles)
{
	float forceMag; 
	float dx, dy, dz, d2, d, minDustDis, minDustDy;
	int myId, yourId, minId;
	
	// You could leave these as globals but putting them as local make the code run faster.
	// Positions are not changed here so it need not be carried forward but forces do need to be 
	// carried forward so it can be used in the move function.
	float posMeX, posMeY, posMeZ, chargeMe, forceMeX, forceMeY, forceMeZ, massMe, ionWakeChargePercentMe, ionWakeLenghtMe;
	
	myId = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ float posYouX[BLOCK], posYouY[BLOCK], posYouZ[BLOCK], chargeYou[BLOCK], ionWakeChargePercentYou[BLOCK], ionWakeLenghtYou[BLOCK];
	
	// Making sure we are not out working past the number of particles.
	if(myId < numberOfParticles)
	{
		// Setting up the local variables we defined above.
		posMeX = dustPos[myId].x;
		posMeY = dustPos[myId].y;
		posMeZ = dustPos[myId].z;
		chargeMe = dustPos[myId].w;
		ionWakeChargePercentMe = ionWake[myId].x;
		ionWakeLenghtMe = ionWake[myId].y;
		forceMeX = 0.0f;
		forceMeY = 0.0f;
		forceMeZ = 0.0f;
		massMe = dustForce[myId].w;
		
		minDustDis = cutOffMultiplier*debyeLength;
		minDustDy = 100000.0;
		minId = -1;
		ionWakeInfoGPU[myId].d = 10000.0;
		ionWakeInfoGPU[myId].dy = 10000.0;
		ionWakeInfoGPU[myId].companionId = -1;
		
		// Getting dust to dust (ashes to ashes) forces and the dust to ionWake forces. Also making calculations on the ionwake.
		for(int j = 0; j < gridDim.x; j++)
		{
			// This puts a whole block of positions in shared memory so they can be accessed quickly (Also the charge).
			posYouX[threadIdx.x] 	= dustPos[threadIdx.x + blockDim.x*j].x;
			posYouY[threadIdx.x] 	= dustPos[threadIdx.x + blockDim.x*j].y;
			posYouZ[threadIdx.x] 	= dustPos[threadIdx.x + blockDim.x*j].z;
			chargeYou[threadIdx.x] 	= dustPos[threadIdx.x + blockDim.x*j].w;
			ionWakeChargePercentYou[threadIdx.x] 	= ionWake[threadIdx.x + blockDim.x*j].x;
			ionWakeLenghtYou[threadIdx.x] 	= ionWake[threadIdx.x + blockDim.x*j].y;
			__syncthreads();
	   
			#pragma unroll 32
		    for(int yourSharedId = 0; yourSharedId < blockDim.x; yourSharedId++)	
		    {
		    	yourId = yourSharedId + blockDim.x*j;
		    	// Making sure we are not working on ourself and not past the number of particles.
				if(myId != yourId && yourId < numberOfParticles) 
				{
					//
					dx = posYouX[yourSharedId] - posMeX;
					dy = posYouY[yourSharedId] - posMeY;
					dz = posYouZ[yourSharedId] - posMeZ;
					d2  = dx*dx + dy*dy + dz*dz + 0.000001f;  // Added the 0.000001 so if two dust grains fell on top of each other the code would not crash.
					d  = sqrt(d2);
					
					// For a coulombic force use this.
					//forceMag = -coulombConstant*chargeYou[yourSharedId]*chargeMe/d2;
					
					// For a Yukawa force use this. A Coulombic force only takes into account regular electrons or regular ions interacting with one another. A Yukawa force takes into account that an ion or dust grain or whatever may have electrons and stuff floating around it which shields it from interacting with other particles after a certain distance. Because it takes all this extra stuff into account, it's more accurate and we want to use it instead of the Coulombic force.
					forceMag = (-coulombConstant*chargeYou[yourSharedId]*chargeMe/d2)*(1.0f + d/debyeLength)*exp(-d/debyeLength);
					
					forceMeX += forceMag*dx/d;
					forceMeY += forceMag*dy/d;
					forceMeZ += forceMag*dz/d;
					
					// Finding the closest dust that is below the current dust and within debyeLengthMultiplier*debyeLengths.
					// We will use this to set the ionWake of the two dusts in question.
					// This will be done in the move function to remove any race conditions.
					if(dy < 0.0f) // If dy is negative you are below me.
					{ 
						if(d < minDustDis)
						{
							minDustDis = d;
							minDustDy = dy;
							minId = yourId;  // This needs to be the real Id not what is in shared memory.
						}
					}
					
					// Adding on the force caused by every other ionWake to the me dust. This has a debye length in it I'm not sure if this is true.
					dy = (posYouY[yourSharedId] - ionWakeLenghtYou[yourSharedId]) - posMeY;
					d2  = dx*dx + dy*dy + dz*dz + 0.000001f;  // Added the 0.000001 so if two dust grains fell on top of each other the code would not crash.
					d  = sqrt(d2);
					
					forceMag = (coulombConstant*chargeYou[yourSharedId]*ionWakeChargePercentYou[yourSharedId]*chargeMe/d2)*(1.0f + d/debyeLength)*exp(-d/debyeLength);
					forceMeX += forceMag*dx/d;
					forceMeY += forceMag*dy/d;
					forceMeZ += forceMag*dz/d;
				}
			}
		}
		
		ionWakeInfoGPU[myId].d = minDustDis;   // Saving the minumum distance so you don't have to calculated it again in the move function.
		ionWakeInfoGPU[myId].dy = minDustDy;	// Saving the minumum y distance so you don't have to calculated it again in the move function.
		ionWakeInfoGPU[myId].companionId = minId; // Saving the closest dust's ID so it can be adjusted in the move function.
		
		// Adding on ionWake force from yourself.
		forceMeY += -(chargeMe*ionWakeChargePercentMe*chargeMe/ionWakeLenghtMe)*(1.0 + ionWakeLenghtMe/debyeLength)*exp(-ionWakeLenghtMe/debyeLength);
		
		// Getting dust to bottom plate force.
		// e field is bottomePlatesCharge*(posMeY - sheathHeight). This is the linear force that starts at the sheath. We got this from Dr. Mathews.
		if (posMeY < sheathHeight)
		{
			forceMeY += -chargeMe*bottomePlatesCharge*(posMeY - sheathHeight);
		}
		
		// Getting culomic push back from the cavity.
		d  = sqrt(posMeX*posMeX + posMeZ*posMeZ);
		if (d != 0.0) // If it is zero nothing needs to be done.
		{
			forceMag = -chargeMe*cavityCharge*pow(d/radiusOfCavity,12.0);
			forceMeX += forceMag*posMeX/d;
			forceMeZ += forceMag*posMeZ/d;
		}
		
		// Getting force of gravity
		forceMeY += -1.0*gravity*massMe;
		
		// All the forces have been sumed up for the dust grain so load them up to carry forward to the move function.
		// The mass was not changed. I just loaded it for completeness and just incase it gets changed in the future.
		if(0.001 < posMeY)
		{
			dustForce[myId].x = forceMeX;
			dustForce[myId].y = forceMeY;
			dustForce[myId].z = forceMeZ;
			dustForce[myId].w = massMe;
		}
		else // If the dust grain gets too close or passes through the floor. I put it at the top of the sheath, set its force to zero and set its mass, charge and diameter to the base (maybe it was too heavy).
		{
			dustPos[myId].y = sheathHeight;
			dustPos[myId].w = 1.0;

			dustVel[myId].x = 0.0;
			dustVel[myId].y = 0.0;

			dustVel[myId].z = 0.0;
			dustVel[myId].w = baseDustDiameter;
			
			dustForce[myId].x = 0.0;
			dustForce[myId].y = 0.0;
			dustForce[myId].z = 0.0;
			dustForce[myId].w = 1.0;
			
			printf("\n myId = %d posMeX = %f posMeY = %f posMeZ = %f MassMe = %f, chargeMe = %f\n", myId, posMeX, posMeY, posMeZ, massMe, chargeMe);
		}
	}
}
		
__global__ void moveDust(float4 *dustPos, float4 *dustVel, float4 *dustForce, float4 *ionWake, ionWakeInfoStructure *ionWakeInfoGPU, float baseIonWakeChargePercent, float baseIonWakeLength, float debyeLength, float cutOffMultiplier, float drag, float electronCharge, float baseElectronsPerUnitDiameter, float electronStdPerUnitDiameter, float dt, float time, int numberOfDustParticles)
{
	// Moving the system forward in time with leap-frog and randomly adjusting the charge on each dust particle.
	curandState state;
	float randomNumber;
	float cutOff, reduction;
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	
	// Note: DustForce.w hold the mass of the dust grain.
	if(id < numberOfDustParticles)
	{
		// Updating my ionwake percent charge and length below the dust if its distance to the its nearest dust grain is within debyeLengthMultiplier*debyeLengths.
		// Also adding the percent charge that the upstream ionWake lost to the down stream ionWake.
		// Need to do this before you update the positions or you will get a miss read on dy.
		if(ionWakeInfoGPU[id].dy < 0.0) // It was initialized to 100000.0 and if no dust grain is close enough it will stay -100000.0.
		{
			cutOff = cutOffMultiplier*debyeLength;
			// This is a quadratic function that goes from 1 to 0 as the dust-dust distance goes from debyeLengthMultiplier*debyeLength to zero.
			// Using a second order because as the bottom dust ets close to the top dust it will be eating up a ring (second order) of ions that would have added to the ionwake.
			// It will be used to decrease the top dust's ionwake and give this lose to the bottom dust's ionwake.
			reduction = (1.0f - ionWakeInfoGPU[id].d*ionWakeInfoGPU[id].d/(cutOff*cutOff))*(ionWakeInfoGPU[id].dy/ionWakeInfoGPU[id].d)*(ionWakeInfoGPU[id].dy/ionWakeInfoGPU[id].d);
			// Decreasing the top charge
			ionWake[id].x = baseIonWakeChargePercent - baseIonWakeChargePercent*reduction;
			// Increasing the bottom charge
			ionWake[ionWakeInfoGPU[id].companionId].x = baseIonWakeChargePercent + baseIonWakeChargePercent*reduction;
			
			// This is a linear that goes from 1 to 0 as the dust-dust distance goes from debyeLengthMultiplier*debyeLength to zero.
			// Using a first order because the as the bottom dust moves up linearly it will displace a ring of ions that would have added to the ionwake.
			// It will be used to decrease the top dust's ionwake length below the top dust.
			//reduction = ionWake[id].z/cutOff;
			reduction = (1.0f - ionWakeInfoGPU[id].d/(cutOff))*(ionWakeInfoGPU[id].dy/ionWakeInfoGPU[id].d)*(ionWakeInfoGPU[id].dy/ionWakeInfoGPU[id].d);
			ionWake[id].y = baseIonWakeLength - baseIonWakeLength*reduction;
		}
		else
		{
			// If for some reason the ionwake didn't get turned back on it is reset here.
			ionWake[id].x = baseIonWakeChargePercent;
			ionWake[id].y = baseIonWakeLength;
		}	
		
		if(time == 0.0)
		{
			dustVel[id].x += 0.5f*dt*(dustForce[id].x - drag*dustVel[id].x)/dustForce[id].w;
			dustVel[id].y += 0.5f*dt*(dustForce[id].y - drag*dustVel[id].y)/dustForce[id].w;
			dustVel[id].z += 0.5f*dt*(dustForce[id].z - drag*dustVel[id].z)/dustForce[id].w;
		}
		else
		{
			dustVel[id].x += dt*(dustForce[id].x - drag*dustVel[id].x)/dustForce[id].w;
			dustVel[id].y += dt*(dustForce[id].y - drag*dustVel[id].y)/dustForce[id].w;
			dustVel[id].z += dt*(dustForce[id].z - drag*dustVel[id].z)/dustForce[id].w;
		}

		dustPos[id].x += dustVel[id].x*dt;
		dustPos[id].y += dustVel[id].y*dt;
		dustPos[id].z += dustVel[id].z*dt;
		
		// Randomly perturbating the dust electron count. 
		// This gets a little involved. I first get a standard normal distributed number (Mean 0 StDev 1).
		// Then I set its StDev to the number of electrons that fluctuate per unit dust diameter for this dust grain size.
		// Then I set the mean to how much above or below the base electron per unit dust size.
		// ie. if it has more than it should it has a higher prob of losing and vice versa if it has less than it should.
		// This is just what I came up with and it could be wrong but below is how I did this.
		// dustPos.w carries the charge and dustVel.w carries the diameter.
		
		// Initailizing the cudarand function.
		curand_init(clock64(), id, 0, &state);
		// This gets a random number with mean 0.0 and stDev 1.0;.
		randomNumber = curand_normal(&state);
		// This sets the electron fluctuation for this sized dust grain and makes it the stDev.
		randomNumber *= electronStdPerUnitDiameter*dustVel[id].w;
		
		// This has a mean of zero which would just create a random walk. I don't think this is what you want.
		// Dust grains with more electrons than they should have should in general loose electrons 
		// and those with less than they should should in general gain more electrons.
		// We will accomplish this by setting the mean to be the oposite of how much above or below 
		// the base amount you are at this time.
		// This works out to be base number - present number
		randomNumber += baseElectronsPerUnitDiameter*dustVel[id].w - dustPos[id].w/electronCharge;
		
		// Now add/subtract this number of electron to the existing charge.
    	dustPos[id].w += randomNumber*electronCharge;
	   
    	// If the amount of charge ends up being negative which probablistically it could, set it to zero
    	if(dustPos[id].w < 0.0) dustPos[id].w = 0.0;
	}				
}

void n_body()
{	
	if(Pause != 1)
	{	
		getForces<<<Grid, Block>>>(DustPositionGPU, DustVelocityGPU, DustForceGPU, IonWakeGPU, IonWakeInfoGPU, BaseDustDiameter, CoulombConstant, DebyeLength, CutOffMultiplier, RadiusOfCavity, CavityCharge, HieghtOfCavity, SheathHeight, BaseIonWakeChargePercent, BottomePlatesCharge, Gravity, NumberOfDustParticles);
		moveDust <<<Grid, Block>>>(DustPositionGPU, DustVelocityGPU, DustForceGPU, IonWakeGPU, IonWakeInfoGPU, BaseIonWakeChargePercent, BaseIonWakeLength, DebyeLength, CutOffMultiplier, Drag, ElectronCharge, BaseElectronsPerUnitDiameter, electronStdPerUnitDiameter, Dt, RunTime, NumberOfDustParticles);
				
		DrawTimer++;
		if(DrawTimer == DrawRate) 
		{
			cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy DustPositionCPU down");
			cudaMemcpy( IonWakeCPU, IonWakeGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy IonWakeCPU down");
			drawPicture();
			DrawTimer = 0;
		}
		
		PrintTimer++;
		if(PrintTimer == PrintTimeRate) 
		{
			system("clear");
			printf("Total run time = %f seconds\n", RunTime*TimeUnit);
			printf("Bottom plate charge: %f\n", BottomePlatesCharge);
			printf("Cavity charge: %f\n", CavityCharge);
			printf("Drag: %f\n", Drag);
			printf("BaseIonWakeCharge: %f\n", BaseIonWakeChargePercent);
			PrintTimer = 0;
		}
		
		RunTime += Dt;
	}
	else
	{
		drawPicture(); // This looks wierd but I had to do it so I could work on the view while it is paused. Consequest of the idle callback calling n_body().
	}
}

void setup()
{	
	readSimulationParameters();
	allocateMemory();
	setUnitConvertions();
	PutConstantsIntoOurUnits();
	setInitialConditions();
	
	// Coping up to GPU
	cudaMemcpy( DustPositionGPU, DustPositionCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
	errorCheck("cudaMemcpy DustPositionCPU up");
    cudaMemcpy( DustVelocityGPU, DustVelocityCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
    errorCheck("cudaMemcpy DustVelocityGPU up");
    cudaMemcpy( DustForceGPU, DustForceCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
    errorCheck("cudaMemcpy DustForceGPU up");
    cudaMemcpy( IonWakeGPU, IonWakeCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
    errorCheck("cudaMemcpy IonWakeGPU up");
    cudaMemcpy( IonWakeInfoGPU, IonWakeInfoCPU, NumberOfDustParticles*sizeof(ionWakeInfoStructure), cudaMemcpyHostToDevice );
    errorCheck("cudaMemcpy IonWakeGPU up");
    
	DrawTimer = 0;
	PrintTimer = 0;
	RunTime = 0.0;
	Pause = 1;
	MovieOn = 0;
	Trace = 0;
	LaserOn = 0;
	DrawBottomRing = 1;
	MouseOn = 0;
	TranslateRotate = 0;
}

void errorCheck(const char *message)
{
  cudaError_t  error;
  error = cudaGetLastError();

  if(error != cudaSuccess)
  {
    printf("\n CUDA ERROR: %s = %s\n", message, cudaGetErrorString(error));
    exit(0);
  }
}

void processMainMenuEvents(int option);
void processSubMenuEvents(int option);
int mainMenu;
int unPauseMenu;
void createGLUTMenus() {
	// You must create the submenu first so you have its id.
	int viewSubMenu = glutCreateMenu(processSubMenuEvents);
	mainMenu = glutCreateMenu(processMainMenuEvents);
	// attach the menu to the right button
	glutAttachMenu(GLUT_RIGHT_BUTTON);

	/* Any time we want to add a menu item to to the main menu that is a submenu, use the glutAddSubMenu function, pass the string and id of the menu created (you must create it above like a regular menu).
	*/
	glutAddSubMenu("Change view", viewSubMenu);
	glutAddMenuEntry("Toggle Bottom Ring Visibility", BOTTOM_RING);

	// Lets set the id to the submenu.
	glutSetMenu(viewSubMenu);
	glutAddMenuEntry("Top View - (o)", TOP_VIEW);
	glutAddMenuEntry("Side View - (O)", SIDE_VIEW);
	glutAddMenuEntry("Frustrum View - (f)", FRUSTRUM);
	glutAddMenuEntry("Orthographic View - (F)", ORTHO);

	//unPauseMenu = glutCreateMenu(processMainMenuEvents);

}

void processMainMenuEvents(int option) 
{
	switch (option) 
	{
		case BOTTOM_RING:
			KeyPressed('b', 0, 0);
			break;
	}
}

void processSubMenuEvents(int option)
{
	switch (option) 
	{
		case TOP_VIEW:
			KeyPressed('o', 0, 0);
			break;
		case SIDE_VIEW:
			KeyPressed('O', 0, 0);
			break;
		case FRUSTRUM:
			KeyPressed('f', 0, 0);
			break;
		case ORTHO:
			KeyPressed('F', 0, 0);
			break;
	}
}

int main(int argc, char** argv)
{
	setup();
	
	XWindowSize = 1000;
	YWindowSize = 1000; 

	// Clip plains
	Near = 0.2;
	Far = 80.0*RadiusOfCavity;

	//Direction here your eye is located location
	EyeX = 0.0*RadiusOfCavity;
	EyeY = 0.5*HieghtOfCavity;
	EyeZ = 3.0*RadiusOfCavity;
	
	//EyeX = 0.01*RadiusOfCavity;
	//EyeY = HieghtOfCavity;
	//EyeZ = 0.0*RadiusOfCavity;

	//Where you are looking
	CenterX = 0.0;
	CenterY = 0.5*HieghtOfCavity;
	CenterZ = 0.0;
	CenterOfView.x = CenterX;
	CenterOfView.y = CenterY;
	CenterOfView.z = CenterZ;

	//Up vector for viewing
	UpX = 0.0;
	UpY = 1.0;
	UpZ = 0.0;
	
	AngleOfView.x = 0.0;
	AngleOfView.y = 0.0;
	AngleOfView.z = 0.0;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(5,5);
	Window = glutCreateWindow("Calvin and Hobbs");
	
	gluLookAt(EyeX, EyeY, EyeZ, CenterX, CenterY, CenterZ, UpX, UpY, UpZ);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, Near, Far);
	glMatrixMode(GL_MODELVIEW);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mymouse);
	glutKeyboardFunc(KeyPressed);
	glutIdleFunc(idle);
	createGLUTMenus(); // This creates the right-click menus.
	glutMainLoop();
	return 0;
}






/*
	Dust parameters:
	mass =   	7.70000e-13 	kg
	charge =  	-3.20000e-15  	Coulombs

	radius = 	4.90000e-06 	m

	#Parabolic potential...radial confinement
	Include parabolic potential?	0			!0=NO, 1=YES
	Include polynomial potential?	1			!0=NO, 1=YES--overrides parabolic potential
	Eigenfrequency squared			0.5685   	!Eigenfrequency for the potential
	Radial extent of crystal		0.063     	!in meters. For use with polynomial pot


	#Plasma and gas parameters
	Ion plasma density				1e14		!Number density 1/m^3
	Ion mass						6.4e-27		!Ion mass in kg
	Temperature						300			!In K
	Pressure						60.0		!Gas pressure in Pa


	// Height above lower electrode defined to be at z = 0

	double height = pos[Z];
	double height_sq = pos[Z]*pos[Z];

	// Linear electric field – set by experimentally observed levitation heights
	// of 8.89 mf particles for various sheath widths (power/pressure dependent)
	// f[Z] -= -k*fabs(charge)*(box_size/2.0 - pos[Z])/mass + g;
	// above the sheath height, the electric field is zero
	  
	if(pos[Z] < .0106)

	{
		e_field = 2 * 3.44e5* (pos[Z] - .0106); //for 11mm sheath
	}
	
	if(pos[Z] < .0071)
	{

		e_field = 2 * 1.4e6 * (pos[Z] - .0071); //for 7mm sheath
	}
	
	if(pos[Z] < .00322)
	{
		e_field = 2 * 6.75e6 * (pos[Z] - .00322); //for 3mm sheath

	}
	else
	{
		e_field = 0;
	}


	// 5th order polynomial e-field in z based on fluid models of GEC cell
	// These values for a specific V_pp and pressure set in the model.
	if (pos[Z] < 0)
	{

		e_field = -8083; //particles can't really go below electrode
	}
	else if (pos[Z] > .0254)
	{
		e_field = 8083; //dust can't go above upper electrode
	}
	else
	{
    	e_field = -8083 + 553373*height + 2.0e8*height_sq + -3.017e10*height*height_sq + 1.471e12*height_sq*height_sq + -2.306e13*height*height_sq*height_sq;
    }

    
	f[Z] += charge * e_field/mass - 9.81;
	
	For the 10th-order polynomial radial confinement


	// Radial Polynomial Potential function:  Lorin Matthews
	// high order polynomial in radial distance to give a very
	// flat confinement out to the 'edge' of the 24" electrode
	// Used for simulation of Cell 3                        
	// 09-16-2020

	void bt610_PolyPot(VECTOR f,VECTOR f_dot, VECTOR f_2dot, VECTOR f_3dot, VECTOR pos, double charge, int order )

	{

	// Based on potential used by Meyer, Laut, et al. PRL, 119, 255001, 2017
	//  V_i = 0.5M(Omega_h^2 rho_i^10/R^8 + Omega_z^2 z_i^2)
	//  where rho = sqrt(x^2 + y^2)

	//  M is the mass of the particles
	//  Omega_h and Omega_z are horizontal and vertical confining freqs
	//  R is approximate horizontal radius of the crystal.
	//  In this case, we only need the radial part of the confinement.
	//  In Meyer et al., M = 0.61e-12 kg, Q = -24648e, Omega_h=2pi*.12 rad/s
	//  Omega_z = 2pi*14 rad/s, R = 63 mm


	double rho, rho_sq,rho_8,rad_force, rho3;
	double R8_inv =  bt030_RunPar.srpr_R8_inv; //=1/R^8
	double omega_h_sq = bt030_RunPar.srpr_eigenfreq2; //2 * BT010_PI * 0.12;

	rho_sq = pos[X]*pos[X] + pos[Y]*pos[Y];
	rho_8 = rho_sq*rho_sq*rho_sq*rho_sq;

	rad_force = 0.5*omega_h_sq*rho_8*R8_inv;
	eigenfreq2 and R8 calculated from being read in from parameter file:
	if (ptr->srpr_include_para_poten || ptr-> srpr_include_poly_poten)
	{

		ReadDbl("Eigenfrequency squared", &ptr->srpr_eigenfreq2);
	}
	else
	{
		ptr->srpr_eigenfreq2 = 0.0;

	}
	
	if (ptr-> srpr_include_poly_poten)
	{
		ReadDbl("Radial extent of crystal",&dum_dbl);

		dum_dbl2 = dum_dbl*dum_dbl; //R^2
		dum_dbl2 = dum_dbl2*dum_dbl2; //R^4
		dum_dbl2 = dum_dbl2*dum_dbl2; //R^8
		ptr->srpr_R8_inv = 1.0/dum_dbl2;
	}

	else
	{
		ptr->srpr_R8_inv = 0.0;
	}

	Unfortunately, the Kodiak meltdown this summer ate my parameter files where I had tested the input values for this option.  The one thing I would change here is that the “eigenfrequency” set by Omega_h includes the effect of charge – it would be better to let this change as the charge on the particles varies.

	I will send you picture of my notes in my research notebook.  And I have a note that there is a repository on GitHub, so I will see if a parameter file was saved there.
	
*/






// -----------------------------------------------------------------------------------------------------------------------------------------------------------


//                   Code copied from here down is stuff I copied from a demo on GTK+. Going to try to connect things together.


// -----------------------------------------------------------------------------------------------------------------------------------------------------------

/*      This is how you compile the gtk program. Not sure how to combine with Cuda yet though. And also, how to incorporate Glut in this.
	gcc `pkg-config --cflags gtk+-3.0` -o modernGTK modernGTK.c `pkg-config --libs gtk+-3.0`
	./modernGTK

	
	This simple application demonstrates how to create a simple GTK application using more modern techniques.
	Some things differ in here from simple.c to reflect GTK3 instead of GTK2.
	For example, it's a little more clear how the layout works.
*/



/*static void print_hello(GtkWidget *widget, gpointer data) {
	g_print("Hello World\n"); // print to a terminal if the application was started from one.
}

static void activate(GtkApplication *app, gpointer user_data) 
{
	// GtkWidget is the base class that all widgets in GTK+ derive from. Manages widget lifestyle, states, and style.
	GtkWidget *window;
	GtkWidget *button;
	GtkWidget *button_box;
	GtkWidget *glBox;
	
	/* Actually creates the window which is a GTKWindow, a toplevel window that can contain other
	   widgets. The window type is GTK_WINDOW_TOPLEVEL, so it has a titlebar and border (what we typically want).
	*/
	//window = gtk_application_window_new(app);
	
	/* -------- Customizing a few things: setting title, changing size of window, entering the window on the screen --------*/
	
	// doing this: GTK_WINDOW(window) casts the window (which is a pointer to a GtkWidget object) to a GtkWindow - GTK_WINDOW is a macro.
	
	//gtk_window_set_title(GTK_WINDOW(window), "GTK Tutorial");	// specify window with GTK_WINDOW(window), and pass title to display
	//gtk_window_set_default_size(GTK_WINDOW(window), 800, 500);	// specify window with GTK_WINDOW(window), and pass width, height
	//gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER);
	
	// Decide on layout of the button
	/*button_box = gtk_button_box_new(GTK_ORIENTATION_HORIZONTAL);
	gtk_container_add(GTK_CONTAINER(window), button_box);
	
	button = gtk_button_new_with_label("Hello World");
	g_signal_connect(button, "clicked", G_CALLBACK(print_hello), NULL);  /* print_hello is the event handler, NULL because print_hello 
										doesn't take any data.*/
	/*g_signal_connect_swapped(button, "clicked", G_CALLBACK(gtk_widget_destroy), window);
	// the swapped version of g_signal,_connect allows the callback function to take a parameter passed in as data.
	gtk_container_add(GTK_CONTAINER(button_box), button); /* this will actually add the button to the window (technically the button_box, 
								 but the button_box contains the button */
	
	//gtk_widget_show_all(window);
//}*/

// This is the equivalent of the glutDisplayFunc() callback function. So just draw stuff inside of here.
//static gboolean
//render (GtkGLArea *area, GdkGLContext *context)
//{
  // inside this function it's safe to use GL; the given
  // GdkGLContext has been made current to the drawable
  // surface used by the `GtkGLArea` and the viewport has
  // already been set to be the size of the allocation

  // we can start by clearing the buffer
  //glClearColor (0, 0, 0, 0);
  //glClear (GL_COLOR_BUFFER_BIT);

  // draw your object
  // draw_an_object ();
  
//  Display();

  // we completed our drawing; the draw commands will be
  // flushed at the end of the signal emission chain, and
  // the buffers will be drawn on the window
  
//  return TRUE;
//}

//void setup_glarea (void)
//{
  // create a GtkGLArea instance
  //GtkWidget *gl_area = gtk_gl_area_new ();

  // connect to the "render" signal
 // g_signal_connect (gl_area, "render", G_CALLBACK (render), NULL);
//}*/

/*int main(int argc, char *argv[]) 
{

	GtkApplication *app;
	int status;
	
	app = gtk_application_new("edu.tarleton.pmg.complex-plasmas", G_APPLICATION_FLAGS_NONE);	// create a new application (just a container to hold everything)
	g_signal_connect(app, "activate", G_CALLBACK(activate), NULL); // This will cause the activate function we created to be called
	status = g_application_run(G_APPLICATION(app), argc, argv);
	g_object_unref(app); // Tidy up and free the memory when we are through.

	return 0;
}*/

// ---------------------


	
	
	
	

 


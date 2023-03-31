void Display()
{
	drawPicture();
}

void idle()
{
	n_body();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);
}

/*	Explanation of some variables.
	Pause, 0 means we are not paused, 1 means we are.

*/


void KeyPressed(unsigned char key, int x, int y)
{
	float dx = 0.2;
	float dy = 0.2;
	float dz = 0.2;
	float angle = 0.4;
	double diameter, radius, mass, charge, electronsPerUnitDiameter, numberOfElectrons;
	
	if(key == 'q')
	{
		glutDestroyWindow(Window);
		printf("\nw Good Bye\n");
		exit(0);
	}
	if(key == 'o') // Top view
	{
		glLoadIdentity();
		glTranslatef(0.0, -CenterY, 0.0);
		glTranslatef(0.0, 0.0, -RadiusOfCavity);
		glTranslatef(CenterOfView.x, CenterOfView.y, CenterOfView.z);
			glRotatef(90.0, 1.0, 0.0, 0.0);
		glTranslatef(-CenterOfView.x, -CenterOfView.y, -CenterOfView.z);
			
		//glOrtho(-RadiusOfCavity, RadiusOfCavity, -RadiusOfCavity, RadiusOfCavity, Near, Far);
		//glMatrixMode(GL_MODELVIEW);
		drawPicture();
	}
	if(key == 'O') // Side view
	{
		glLoadIdentity();
		glTranslatef(0.0, -CenterY, 0.0);
		glTranslatef(0.0, 0.0, -RadiusOfCavity);
		drawPicture();
	}
	if(key == 'f') // Frustrum view
	{
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glFrustum(-0.2, 0.2, -0.2, 0.2, Near, Far);
		glMatrixMode(GL_MODELVIEW);
		drawPicture();
	}
	if (key == 'F') { // Orthogonal view
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		// glOrtho(-0.2, 0.2, -0.2, 0.2, Near, Far);
		glOrtho(-RadiusOfCavity, RadiusOfCavity, -RadiusOfCavity, RadiusOfCavity, Near, Far);
		glMatrixMode(GL_MODELVIEW);
		drawPicture();
	}
	
	if (key == 'p') // Now, p just toggles the simulation on and off.
	{
		if (Pause == 0) {
			Pause = 1;
			printf("\033[1;31m");
			printf("\nPaused\n");
			printf("\033[0m");
		} else
		{
			Pause = 0;
			printf("\033[1;32m");
			printf("\nUnpaused\n");
			printf("\033[0m");
		}
	}
	/*if(key == 'p') // unpause
	{
		Pause = 1;
	}
	if(key == 'P')	// pause
	{
		Pause = 0;
	}*/
	
	if(key == 'r')
	{
		TranslateRotate = 1;
	}
	if(key == 'R')
	{
		TranslateRotate = 0;
	}
	
	if(key == 'm')
	{
		// Setting up the movie buffer.
		const char* cmd = "ffmpeg -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
		              "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip video.mp4";
		MovieFile = popen(cmd, "w");
		//Buffer = new int[XWindowSize*YWindowSize];
		Buffer = (int*)malloc(XWindowSize*YWindowSize*sizeof(int));
		MovieOn = 1;
	}
	if(key == 'M')
	{
		if(MovieOn == 1) 
		{
			pclose(MovieFile);
		}
		free(Buffer);
		MovieOn = 0;
	}
	
	if(key == 's')
	{	
		int pauseFlag;
		FILE* ScreenShotFile;
		int* buffer;
		const char* cmd = "ffmpeg -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
		              "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output1.mp4";
		ScreenShotFile = popen(cmd, "w");
		buffer = (int*)malloc(XWindowSize*YWindowSize*sizeof(int));
		
		if(Pause == 0) 
		{
			Pause = 1;
			pauseFlag = 0;
		}
		else
		{
			pauseFlag = 1;
		}
		
		for(int i =0; i < 1; i++)
		{
			drawPicture();
			glReadPixels(5, 5, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
			fwrite(buffer, sizeof(int)*XWindowSize*YWindowSize, 1, ScreenShotFile);
		}
		
		pclose(ScreenShotFile);
		free(buffer);
		system("ffmpeg -i output1.mp4 screenShot.jpeg");
		system("rm output1.mp4");
		
		Pause = pauseFlag;
		//ffmpeg -i output1.mp4 output_%03d.jpeg
	}
	
	if(key == 't')
	{
		Trace = 1;
	}
	if(key == 'T')
	{
		Trace = 0;
	}
	
	if(key == 'b')
	{
		DrawBottomRing = 1;
		drawPicture();
	}
	if(key == 'B')
	{
		DrawBottomRing = 0;
		drawPicture();
	}
	
	if(key == 'l')
	{
		LaserOn = 0;
	}
	if(key == 'L')
	{
		LaserOn = 1;
	}
	
	if(key == 'a')
	{
		MouseOn = 1;
		Pause = 1;
	}
	if(key == 'A')
	{
		MouseOn = 0;
		Pause = 0;
	}
	
	if(key == 'i')
	{
		BaseIonWakeChargePercent -= 0.01;
		if(BaseIonWakeChargePercent < 0.0) BaseIonWakeChargePercent = 0.0;
		for(int i = 0; i < NumberOfDustParticles; i++)
		{
			IonWakeCPU[i].x = BaseIonWakeChargePercent;
		}
		printf("\n BaseIonWakeChargePercent = %f \n", BaseIonWakeChargePercent);
		cudaMemcpy( IonWakeGPU, IonWakeCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
    	errorCheck("cudaMemcpy IonWakeGPU up");
	}
	if(key == 'I')
	{
		BaseIonWakeChargePercent += 0.01;
		for(int i = 0; i < NumberOfDustParticles; i++)
		{
			IonWakeCPU[i].x = BaseIonWakeChargePercent;
		}
		printf("\n BaseIonWakeChargePercent = %f \n", BaseIonWakeChargePercent);
		cudaMemcpy( IonWakeGPU, IonWakeCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
    	errorCheck("cudaMemcpy IonWakeGPU up");
	}
	
	
	//EyeX = 2.0*RadiusOfCavity;
	//EyeY = 0.5*HeightOfCavity;
	//EyeZ = 2.0*RadiusOfCavity;
	
	//Where you are looking
	//CenterX = 0.0;
	//CenterY = 0.5*HeightOfCavity;
	//CenterZ = 0.0;
	
	//CenterOfView.x = EyeX;
	//CenterOfView.y = EyeY;
	//CenterOfView.z = EyeZ;
	
	// Adjusting power on lower plate
	double deltaPowerBottomPlate = 1.0e8;
	if(key == 'V')
	{
		deltaPowerBottomPlate = deltaPowerBottomPlate*TimeUnit*TimeUnit*ChargeUnit/MassUnit;
		BottomPlateConstant += deltaPowerBottomPlate;
		printf("\nBottomePlatesCharge = %e", BottomPlateConstant/(TimeUnit*TimeUnit*ChargeUnit/MassUnit));
	}
	if(key == 'v')
	{
		deltaPowerBottomPlate = deltaPowerBottomPlate*TimeUnit*TimeUnit*ChargeUnit/MassUnit;
		BottomPlateConstant -= deltaPowerBottomPlate;
		if(BottomPlateConstant < 0.0) BottomPlateConstant = 0.0;
		printf("\nBottomePlatesCharge = %e", BottomPlateConstant/(TimeUnit*TimeUnit*ChargeUnit/MassUnit));
	}
	
	// Adjusting power on side of cavity
	double deltaCavityConfinementConstant = 1.0e5;
	if(key == 'C')
	{
		deltaCavityConfinementConstant = deltaCavityConfinementConstant*TimeUnit*TimeUnit*ChargeUnit/MassUnit;
		CavityConfinementConstant += deltaCavityConfinementConstant;
		printf("\nCavityConfinementConstant = %e", CavityConfinementConstant/(TimeUnit*TimeUnit*ChargeUnit/MassUnit));
	}
	if(key == 'c')
	{
		deltaCavityConfinementConstant = deltaCavityConfinementConstant*TimeUnit*TimeUnit*ChargeUnit/MassUnit;
		CavityConfinementConstant -= deltaCavityConfinementConstant;
		if(CavityConfinementConstant < 0.0) CavityConfinementConstant = 0.0;
		printf("\nCavityConfinementConstant = %e", CavityConfinementConstant/(TimeUnit*TimeUnit*ChargeUnit/MassUnit));
	}
	
	// Adjusting drag or pressure
	double deltaDrag = 0.01;
	if(key == 'D')
	{
		Drag += deltaDrag;
		printf("\nDrag = %f", Drag);
	}
	if(key == 'd')
	{
		Drag -= deltaDrag;
		if(Drag < 0.0) Drag = 0.0;
		printf("\nDrag = %f", Drag);
	}
	
	if(key == 'x')
	{
		if(TranslateRotate == 0) 
		{
			glTranslatef(dx, 0.0, 0.0);
			CenterOfView.x += dx;
		}
		else 
		{
			glTranslatef(CenterOfView.x, CenterOfView.y, CenterOfView.z);
			glRotatef(angle, 1.0, 0.0, 0.0);
			glTranslatef(-CenterOfView.x, -CenterOfView.y, -CenterOfView.z);
			AngleOfView.x += angle;
		}
		drawPicture();
	}
	if(key == 'X')
	{
		if(TranslateRotate == 0) 
		{
			glTranslatef(-dx, 0.0, 0.0);
			CenterOfView.x += -dx;
		}
		else 
		{
			glTranslatef(CenterOfView.x, CenterOfView.y, CenterOfView.z);
			glRotatef(-angle, 1.0, 0.0, 0.0);
			glTranslatef(-CenterOfView.x, -CenterOfView.y, -CenterOfView.z);
			AngleOfView.x += -angle;
		}
		drawPicture();
	}
	
	if(key == 'y')
	{
		if(TranslateRotate == 0) 
		{
			glTranslatef(0.0, dy, 0.0);
			CenterOfView.x += dy;
		}
		else 
		{
			glTranslatef(CenterOfView.x, CenterOfView.y, CenterOfView.z);
			glRotatef(angle, 0.0, 1.0, 0.0);
			glTranslatef(-CenterOfView.x, -CenterOfView.y, -CenterOfView.z);
			AngleOfView.y += angle;
		}
		
		drawPicture();
	}
	if(key == 'Y')
	{
		if(TranslateRotate == 0) 
		{
			glTranslatef(0.0, -dy, 0.0);
			CenterOfView.x += -dy;
		}
		else 
		{
			glTranslatef(CenterOfView.x, CenterOfView.y, CenterOfView.z);
			glRotatef(-angle, 0.0, 1.0, 0.0);
			glTranslatef(-CenterOfView.x, -CenterOfView.y, -CenterOfView.z);
			AngleOfView.y += -angle;
		}
		drawPicture();
	}
	
	if(key == 'z')
	{
		if(TranslateRotate == 0) 
		{
			glTranslatef(0.0, 0.0, dz);
			CenterOfView.x += dz;
		}
		else 
		{
			glTranslatef(CenterOfView.x, CenterOfView.y, CenterOfView.z);
			glRotatef(angle, 0.0, 0.0, 1.0);
			glTranslatef(-CenterOfView.x, -CenterOfView.y, -CenterOfView.z);
			AngleOfView.z += angle;
		}
		drawPicture();
	}
	if(key == 'Z')
	{
		if(TranslateRotate == 0) 
		{
			glTranslatef(0.0, 0.0, -dz);
			CenterOfView.x += -dz;
		}
		else 
		{
			glTranslatef(CenterOfView.x, CenterOfView.y, CenterOfView.z);
			glRotatef(-angle, 0.0, 0.0, 1.0);
			glTranslatef(-CenterOfView.x, -CenterOfView.y, -CenterOfView.z);
			AngleOfView.z += -angle;
		}
		drawPicture();
	}
	
	if(key == '1')
	{
		cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustPositionCPU down");
		cudaMemcpy( DustVelocityCPU, DustVelocityGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustVelocityCPU down");
		cudaMemcpy( DustForceCPU, DustForceGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustForceCPU down");
		cudaMemcpy( IonWakeCPU, IonWakeGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy IonWakeCPU down");
		
		charge = DustPositionCPU[SelectedDustGrainId1].w;
		diameter = DustVelocityCPU[SelectedDustGrainId1].w;
		numberOfElectrons = charge/ElectronCharge;
		electronsPerUnitDiameter = numberOfElectrons/diameter;
		
		diameter -= 0.0001;
		radius = diameter/2.0;
		mass = DustDensity*(4.0/3.0)*PI*radius*radius*radius;
		charge = ElectronCharge*round(electronsPerUnitDiameter*diameter);
		
		DustVelocityCPU[SelectedDustGrainId1].w = diameter; // Velocity w holds the diameter
		DustForceCPU[SelectedDustGrainId1].w = mass; // Force w holds the mass
		DustPositionCPU[SelectedDustGrainId1].w = charge; // Position holds the charge.
		
		printf("\n Diameter 0f dust grain %d = %f", SelectedDustGrainId1, (DustVelocityCPU[SelectedDustGrainId1].w*LengthUnit)*1.0e6);
		
		// Copying every thing back up to the GPU.
		cudaMemcpy( DustPositionGPU, DustPositionCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustPositionCPU up");
		cudaMemcpy( DustVelocityGPU, DustVelocityCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustVelocityGPU up");
		cudaMemcpy( DustForceGPU, DustForceCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustForceGPU up");
		cudaMemcpy( IonWakeGPU, IonWakeCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy IonWakeGPU up");
		
		drawPicture();
	}
	if(key == '!')
	{
		cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustPositionCPU down");
		cudaMemcpy( DustVelocityCPU, DustVelocityGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustVelocityCPU down");
		cudaMemcpy( DustForceCPU, DustForceGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustForceCPU down");
		cudaMemcpy( IonWakeCPU, IonWakeGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy IonWakeCPU down");
		
		charge = DustPositionCPU[SelectedDustGrainId1].w;
		diameter = DustVelocityCPU[SelectedDustGrainId1].w;
		numberOfElectrons = charge/ElectronCharge;
		electronsPerUnitDiameter = numberOfElectrons/diameter;
		
		diameter += 0.0001;
		radius = diameter/2.0;
		mass = DustDensity*(4.0/3.0)*PI*radius*radius*radius;
		charge = ElectronCharge*round(electronsPerUnitDiameter*diameter);
		
		DustVelocityCPU[SelectedDustGrainId1].w = diameter; // Velocity w holds the diameter
		DustForceCPU[SelectedDustGrainId1].w = mass; // Force w holds the mass
		DustPositionCPU[SelectedDustGrainId1].w = charge; // Position holds the charge.
		
		printf("\n Diameter 0f dust grain %d = %f", SelectedDustGrainId1, (DustVelocityCPU[SelectedDustGrainId1].w*LengthUnit)*1.0e6);
		
		// Copying every thing back up to the GPU.
		cudaMemcpy( DustPositionGPU, DustPositionCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustPositionCPU up");
		cudaMemcpy( DustVelocityGPU, DustVelocityCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustVelocityGPU up");
		cudaMemcpy( DustForceGPU, DustForceCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustForceGPU up");
		cudaMemcpy( IonWakeGPU, IonWakeCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy IonWakeGPU up");
		
		drawPicture();
	}
	
	if(key == '2')
	{
		cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustPositionCPU down");
		cudaMemcpy( DustVelocityCPU, DustVelocityGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustVelocityCPU down");
		cudaMemcpy( DustForceCPU, DustForceGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustForceCPU down");
		cudaMemcpy( IonWakeCPU, IonWakeGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy IonWakeCPU down");
		
		charge = DustPositionCPU[SelectedDustGrainId2].w;
		diameter = DustVelocityCPU[SelectedDustGrainId2].w;
		numberOfElectrons = charge/ElectronCharge;
		electronsPerUnitDiameter = numberOfElectrons/diameter;
		
		diameter -= 0.0001;
		radius = diameter/2.0;
		mass = DustDensity*(4.0/3.0)*PI*radius*radius*radius;
		charge = ElectronCharge*round(electronsPerUnitDiameter*diameter);
		
		DustVelocityCPU[SelectedDustGrainId2].w = diameter; // Velocity w holds the diameter
		DustForceCPU[SelectedDustGrainId2].w = mass; // Force w holds the mass
		DustPositionCPU[SelectedDustGrainId2].w = charge; // Position holds the charge.
		
		printf("\n Diameter 0f dust grain %d = %f", SelectedDustGrainId2, (DustVelocityCPU[SelectedDustGrainId2].w*LengthUnit)*1.0e6);
		
		charge = DustPositionCPU[SelectedDustGrainId3].w;
		diameter = DustVelocityCPU[SelectedDustGrainId3].w;
		numberOfElectrons = charge/ElectronCharge;
		electronsPerUnitDiameter = numberOfElectrons/diameter;
		
		diameter += 0.0001;
		radius = diameter/2.0;
		mass = DustDensity*(4.0/3.0)*PI*radius*radius*radius;
		charge = ElectronCharge*round(electronsPerUnitDiameter*diameter);
		
		DustVelocityCPU[SelectedDustGrainId3].w = diameter; // Velocity w holds the diameter
		DustForceCPU[SelectedDustGrainId3].w = mass; // Force w holds the mass
		DustPositionCPU[SelectedDustGrainId2].w = charge; // Position holds the charge.
		
		printf("\n Diameter 0f dust grain %d = %f", SelectedDustGrainId3, (DustVelocityCPU[SelectedDustGrainId3].w*LengthUnit)*1.0e6);
		
		// Copying every thing back up to the GPU.
		cudaMemcpy( DustPositionGPU, DustPositionCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustPositionCPU up");
		cudaMemcpy( DustVelocityGPU, DustVelocityCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustVelocityGPU up");
		cudaMemcpy( DustForceGPU, DustForceCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustForceGPU up");
		cudaMemcpy( IonWakeGPU, IonWakeCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy IonWakeGPU up");
		
		drawPicture();
	}
	if(key == '@')
	{
		cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustPositionCPU down");
		cudaMemcpy( DustVelocityCPU, DustVelocityGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustVelocityCPU down");
		cudaMemcpy( DustForceCPU, DustForceGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy DustForceCPU down");
		cudaMemcpy( IonWakeCPU, IonWakeGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
		errorCheck("cudaMemcpy IonWakeCPU down");
		
		charge = DustPositionCPU[SelectedDustGrainId2].w;
		diameter = DustVelocityCPU[SelectedDustGrainId2].w;
		numberOfElectrons = charge/ElectronCharge;
		electronsPerUnitDiameter = numberOfElectrons/diameter;
		
		diameter += 0.0001;
		radius = diameter/2.0;
		mass = DustDensity*(4.0/3.0)*PI*radius*radius*radius;
		charge = ElectronCharge*round(electronsPerUnitDiameter*diameter);
		
		DustVelocityCPU[SelectedDustGrainId2].w = diameter; // Velocity w holds the diameter
		DustForceCPU[SelectedDustGrainId2].w = mass; // Force w holds the mass
		DustPositionCPU[SelectedDustGrainId2].w = charge; // Position holds the charge.
		
		printf("\n Diameter 0f dust grain %d = %f", SelectedDustGrainId2, (DustVelocityCPU[SelectedDustGrainId2].w*LengthUnit)*1.0e6);
		
		charge = DustPositionCPU[SelectedDustGrainId3].w;
		diameter = DustVelocityCPU[SelectedDustGrainId3].w;
		numberOfElectrons = charge/ElectronCharge;
		electronsPerUnitDiameter = numberOfElectrons/diameter;
		
		diameter -= 0.0001;
		radius = diameter/2.0;
		mass = DustDensity*(4.0/3.0)*PI*radius*radius*radius;
		charge = ElectronCharge*round(electronsPerUnitDiameter*diameter);
		
		DustVelocityCPU[SelectedDustGrainId3].w = diameter; // Velocity w holds the diameter
		DustForceCPU[SelectedDustGrainId3].w = mass; // Force w holds the mass
		DustPositionCPU[SelectedDustGrainId2].w = charge; // Position holds the charge.
		
		printf("\n Diameter 0f dust grain %d = %f", SelectedDustGrainId3, (DustVelocityCPU[SelectedDustGrainId3].w*LengthUnit)*1.0e6);
		
		// Copying every thing back up to the GPU.
		cudaMemcpy( DustPositionGPU, DustPositionCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice);
		errorCheck("cudaMemcpy DustPositionCPU up");
		cudaMemcpy( DustVelocityGPU, DustVelocityCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustVelocityGPU up");
		cudaMemcpy( DustForceGPU, DustForceCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy DustForceGPU up");
		cudaMemcpy( IonWakeGPU, IonWakeCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
		errorCheck("cudaMemcpy IonWakeGPU up");
		
		drawPicture();
	}
	double deltaPressure = 1;

	if (key == '4') // decrease temperature
	{
		GasPressure -= deltaPressure;
		printf("\n Gas Pressure in millitorr = %f", GasPressure);
		Drag = PressureConstant * GasPressure;
	}
	if (key == '$') // increase temperature
	{
		GasPressure += deltaPressure;
		printf("\n Gas Pressure in millitorr = %f", GasPressure);
		Drag = PressureConstant * GasPressure;
	}
	if (key == 'h') 
	{
	
		FILE *fptr;
  
	    	char filename[100] = "help.txt", c;
	  
	    	// Open file
	    	fptr = fopen(filename, "r");
	    	if (fptr == NULL)
	    	{
			printf("Cannot open file \n");
			exit(0);
		}
		  
		// Read contents from file
		c = fgetc(fptr);
		while (c != EOF)
		{
			printf ("%c", c);
			c = fgetc(fptr);
		}
		 
		fclose(fptr);	
	}
}

void mymouse(int button, int state, int x, int y)
{	
	float myX, myZ;
	float dustX, dustY, dustZ;
	float closest;
	float dx, dy, dz, d;
	double diameter, radius, mass, charge, electronsPerUnitDiameter; // numberOfElectrons, percentOfDustCharge;
	//int SelectedDustGrainId1, SelectedDustGrainId2, SelectedDustGrainId3;
	
	if(state == GLUT_DOWN)
	{
		if(MouseOn == 1)
		{
			// You need to copy everything down from the GPU because the mass is in force.w the diameter is in vel.w and the charge is in pos.w
			cudaMemcpy( DustPositionCPU, DustPositionGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy DustPositionCPU down");
			cudaMemcpy( DustVelocityCPU, DustVelocityGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy DustVelocityCPU down");
			cudaMemcpy( DustForceCPU, DustForceGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy DustForceCPU down");
			cudaMemcpy( IonWakeCPU, IonWakeGPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy IonWakeCPU down");
			
			if(button == GLUT_LEFT_BUTTON)
			{
				//while ((getchar()) != '\n');
				//fflush(stdin);
				myX =  ( 2.0*x/XWindowSize - 1.0)*RadiusOfCavity;
				myZ = -(-2.0*y/YWindowSize + 1.0)*RadiusOfCavity;
				
				// Flashing a big yellow ball where you selected.
				glColor3d(1.0, 1.0, 0.0);
					glPushMatrix();
					glTranslatef(myX, HeightOfCavity/2.0, myZ);
					glutSolidSphere(0.5,20,20);
				glPopMatrix();
				glutSwapBuffers();
				usleep(200000);
				
				// Finding the closest dust grain to where you mouse clicked.
				closest = 10000000.0;
				SelectedDustGrainId1 = -1;
				for(int i = 1; i < NumberOfDustParticles; i++)
				{
					dx = DustPositionCPU[i].x - myX;
					dz = DustPositionCPU[i].z - myZ;
					d = sqrt(dx*dx + dz*dz);
					if(d < closest)
					{
						closest = d;
						SelectedDustGrainId1 = i;
					}
				}
				
				// Setting the color of the selected dust and printing out it's location.
				if(SelectedDustGrainId1 != -1)
				{
					// Coloring the dust grain blue so you know which one you changed.
					DustColor[SelectedDustGrainId1].x = 0.0;
					DustColor[SelectedDustGrainId1].y = 0.0;
					DustColor[SelectedDustGrainId1].z = 1.0;
					dustX = DustPositionCPU[SelectedDustGrainId1].x*LengthUnit;
					dustY = DustPositionCPU[SelectedDustGrainId1].y*LengthUnit;
					dustZ = DustPositionCPU[SelectedDustGrainId1].z*LengthUnit;
					printf("\n\n DustGrain: x = %f, y = %f, z = %f", dustX, dustY, dustZ);
				}
				else
				{
					printf("\n Error no dust grain selected");
					exit(0);
				}
				drawPicture();
				
				/*
				cout << "\n\n******************************************************" << endl;
				cout << " Enter: The diameter of the dust grain in microns." << endl;
				cout << " Enter: Enter -1.0 to leave the diameter unchanged." << endl;
				cout << "******************************************************\n\nDiametr = ";
				
				cin >> diameter;
				if(0.0 < diameter)
				{
					diameter /= 1.0e6;
					diameter /= LengthUnit;
					DustVelocityCPU[SelectedDustGrainId1].w = diameter; // Velocity w holds the diameter
					radius = diameter/2.0;
					mass = DustDensity*(4.0/3.0)*PI*radius*radius*radius;
					DustForceCPU[SelectedDustGrainId1].w = mass; // Force w holds the mass
				}
				*/
			}
			else
			{
				//while ((getchar()) != '\n');
				//fflush(stdin);
				myX =  ( 2.0*x/XWindowSize - 1.0)*RadiusOfCavity;
				myZ = -(-2.0*y/YWindowSize + 1.0)*RadiusOfCavity;
				
				// Flashing a big yellow ball where you selected.
				glColor3d(1.0, 1.0, 0.0);
					glPushMatrix();
					glTranslatef(myX, HeightOfCavity/2.0, myZ);
					glutSolidSphere(0.5,20,20);
				glPopMatrix();
				glutSwapBuffers();
				usleep(200000);
				
				// Finding the closest dust grain to where you mouse clicked.
				closest = 10000000.0;
				SelectedDustGrainId2 = -1;
				for(int i = 1; i < NumberOfDustParticles; i++)
				{
					dx = DustPositionCPU[i].x - myX;
					dz = DustPositionCPU[i].z - myZ;
					d = sqrt(dx*dx + dz*dz);
					if(d < closest)
					{
						closest = d;
						SelectedDustGrainId2 = i;
					}
				}
				
				// Setting the color of the selected dust and printing out it's location.
				if(SelectedDustGrainId2 != -1)
				{
					// Coloring the dust grain blue so you know which one you changed.
					DustColor[SelectedDustGrainId2].x = 0.0;
					DustColor[SelectedDustGrainId2].y = 1.0;
					DustColor[SelectedDustGrainId2].z = 0.0;
					dustX = DustPositionCPU[SelectedDustGrainId2].x*LengthUnit;
					dustY = DustPositionCPU[SelectedDustGrainId2].y*LengthUnit;
					dustZ = DustPositionCPU[SelectedDustGrainId2].z*LengthUnit;
					printf("\n\n Dust grain1: x = %f, y = %f, z = %f", dustX, dustY, dustZ);
				}
				else
				{
					printf("\n Error no dust grain selected");
					exit(0);
				}
				
				// Finding the closest dust grain to the dust grain just selected.
				closest = 10000000.0;
				SelectedDustGrainId3 = -1;
				for(int i = 1; i < NumberOfDustParticles; i++)
				{
					if(i != SelectedDustGrainId2)
					{
						dx = DustPositionCPU[i].x - DustPositionCPU[SelectedDustGrainId2].x;
						dy = DustPositionCPU[i].y - DustPositionCPU[SelectedDustGrainId2].y;
						dz = DustPositionCPU[i].z - DustPositionCPU[SelectedDustGrainId2].z;
						d = sqrt(dx*dx + dy*dy + dz*dz);
						if(d < closest)
						{
							closest = d;
							SelectedDustGrainId3 = i;
						}
					}
				}
				
				// Setting the color of the selected dust and printing out it's location.
				if(SelectedDustGrainId3 != -1)
				{
					// Coloring the dust grain blue so you know which one you changed.
					DustColor[SelectedDustGrainId3].x = 1.0;
					DustColor[SelectedDustGrainId3].y = 0.0;
					DustColor[SelectedDustGrainId3].z = 1.0;
					dustX = DustPositionCPU[SelectedDustGrainId3].x*LengthUnit;
					dustY = DustPositionCPU[SelectedDustGrainId3].y*LengthUnit;
					dustZ = DustPositionCPU[SelectedDustGrainId3].z*LengthUnit;
					printf("\n\n Dust grain2: x = %f, y = %f, z = %f", dustX, dustY, dustZ);
				}
				else
				{
					printf("\n Error no dust grain selected");
					exit(0);
				}
				
				// Setting both selected dust grains to the base diameter, mass, and charge.
				// This will start them at the base for a referance as they change.
				diameter = BaseDustDiameter;
				electronsPerUnitDiameter = BaseElectronsPerUnitDiameter;
				radius = diameter/2.0;
				mass = DustDensity*(4.0/3.0)*PI*radius*radius*radius;
				charge = ElectronCharge*round(electronsPerUnitDiameter*diameter);
				
				DustVelocityCPU[SelectedDustGrainId2].w = diameter; // Velocity w holds the diameter
				DustForceCPU[SelectedDustGrainId2].w = mass; // Force w holds the mass
				DustPositionCPU[SelectedDustGrainId2].w = charge; // Position holds the charge.
				
				DustVelocityCPU[SelectedDustGrainId3].w = diameter; // Velocity w holds the diameter
				DustForceCPU[SelectedDustGrainId3].w = mass; // Force w holds the mass
				DustPositionCPU[SelectedDustGrainId3].w = charge; // Position holds the charge.
			}
			
			drawPicture();
			MouseOn = 0;	
			// Copying every thing back up to the GPU.
			cudaMemcpy( DustPositionGPU, DustPositionCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
			errorCheck("cudaMemcpy DustPositionCPU up");
			cudaMemcpy( DustVelocityGPU, DustVelocityCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
			errorCheck("cudaMemcpy DustVelocityGPU up");
			cudaMemcpy( DustForceGPU, DustForceCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
			errorCheck("cudaMemcpy DustForceGPU up");
			cudaMemcpy( IonWakeGPU, IonWakeCPU, NumberOfDustParticles*sizeof(float4), cudaMemcpyHostToDevice );
			errorCheck("cudaMemcpy IonWakeGPU up");
		}
	}
}


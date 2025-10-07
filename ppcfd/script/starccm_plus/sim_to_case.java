// Simcenter STAR-CCM+ macro: sim_to_case.java
// Written by Simcenter STAR-CCM+ 16.06.010
package macro;

import java.util.*;

import star.common.*;
import star.base.neo.*;

public class sim_to_case extends StarMacro {

  public void execute() {
    execute0();
  }

  private void execute0() {

    Simulation simulation_0 = 
      getActiveSimulation();

    ImportManager importManager_0 = 
      simulation_0.getImportManager();

    // String export_path = "F:\\AI\\data_cases\\" + simulation_0.getPresentationName() + ".case";
    String export_path = "F:\\AI\\wangguan\\case3\\" + simulation_0.getPresentationName() + ".case";

    importManager_0.setExportPath(export_path);

    importManager_0.setFormatType(SolutionExportFormat.Type.CASE);

    importManager_0.setExportParts(new NeoObjectVector(new Object[] {}));

    importManager_0.setExportPartSurfaces(new NeoObjectVector(new Object[] {}));

    Region region_3 = 
      simulation_0.getRegionManager().getRegion("Vehicle");

    Collection<Boundary> mybounds = region_3.getBoundaryManager().getObjects();
    Collection<Boundary> mybounds_filtered = new ArrayList();

    ArrayList<String> ban_list = new ArrayList<>();
    ban_list.add("WT_Ground");
    ban_list.add("WT_Inlet");
    ban_list.add("WT_Outlet");
    ban_list.add("WT_Side");
    ban_list.add("internal-1");
    
    for (Boundary b : mybounds){
      if (ban_list.contains(b.getPresentationName())){
        simulation_0.println("\n\nBaned Wall: " + b.getPresentationName());
      }
      else{
        mybounds_filtered.add(b);
        simulation_0.println("\n\nCollected Wall: " + b.getPresentationName());
      }
    }

    importManager_0.setExportBoundaries(new NeoObjectVector(mybounds_filtered.toArray()));


    importManager_0.setExportRegions(new NeoObjectVector(new Object[] {}));

    PrimitiveFieldFunction p = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("Pressure"));

    PrimitiveFieldFunction wss = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("WallShearStress"));

    PrimitiveFieldFunction area = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("Area"));

    VectorMagnitudeFieldFunction area_magnitude = 
      ((VectorMagnitudeFieldFunction) area.getMagnitudeFunction());

    PrimitiveFieldFunction normal = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("Normal"));

    PrimitiveFieldFunction centroid = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("Centroid"));

    VectorComponentFieldFunction wss0 = 
      ((VectorComponentFieldFunction) wss.getComponentFunction(0));

    VectorComponentFieldFunction wss1 = 
      ((VectorComponentFieldFunction) wss.getComponentFunction(1));

    VectorComponentFieldFunction wss2 = 
      ((VectorComponentFieldFunction) wss.getComponentFunction(2));
    
    VectorComponentFieldFunction normal0 = 
      ((VectorComponentFieldFunction) normal.getComponentFunction(0));

    VectorComponentFieldFunction normal1 = 
      ((VectorComponentFieldFunction) normal.getComponentFunction(1));

    VectorComponentFieldFunction normal2 = 
      ((VectorComponentFieldFunction) normal.getComponentFunction(2));
    
    VectorComponentFieldFunction centroid0 = 
      ((VectorComponentFieldFunction) centroid.getComponentFunction(0));

    VectorComponentFieldFunction centroid1 = 
      ((VectorComponentFieldFunction) centroid.getComponentFunction(1));

    VectorComponentFieldFunction centroid2 = 
      ((VectorComponentFieldFunction) centroid.getComponentFunction(2));

    importManager_0.setExportScalars(new NeoObjectVector(new Object[] {p, wss0, wss1, wss2, area_magnitude, normal0, normal1, normal2, centroid0, centroid1, centroid2}));

    importManager_0.setExportVectors(new NeoObjectVector(new Object[] {}));

    importManager_0.setExportOptionAppendToFile(false);

    importManager_0.setExportOptionDataAtVerts(false);

    importManager_0.setExportOptionSolutionOnly(false);

    importManager_0.export(
      resolvePath(export_path), 
      new NeoObjectVector(new Object[] {}), 
      new NeoObjectVector(mybounds_filtered.toArray()), 
      new NeoObjectVector(new Object[] {}), 
      new NeoObjectVector(new Object[] {}), 
      new NeoObjectVector(new Object[] {p, wss0, wss1, wss2, area_magnitude, normal0, normal1, normal2, centroid0, centroid1, centroid2}), 
      NeoProperty.fromString(
        "{\'exportFormatType\': 2, \'appendToFile\': false, \'solutionOnly\': false, \'dataAtVerts\': false}"
        )
      );
  }
}

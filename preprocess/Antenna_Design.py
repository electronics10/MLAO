import sys
sys.path.append(r"C:\Program Files (x86)\CST STUDIO SUITE 2023\AMD64\python_cst_libraries")
sys.path.append(r"C:\Program Files (x86)\CST STUDIO SUITE 2024\AMD64\python_cst_libraries")
sys.path.append(r"C:\Program Files (x86)\CST STUDIO SUITE 2025\AMD64\python_cst_libraries")
import cst
import cst.results as cstr
import cst.interface as csti
import os
import numpy as np
import difflib


# Design parameter
L = 36 # mm
W = 36 # mm
D = 9 # mm
NX= int(L//D)
NY = int(W//D)
TSTEP = 0.1 # default 0.1 ns for 1~3 GHz
TEND = 3.5 # default duration=3.5 ns for 1~3 GHz

LG = 104
WG = 104
HC = 0.035
HS = 1.6
FEEDX = 5
FEEDY = 0


class CSTInterface:
    def __init__(self, fname):
        self.full_path = os.getcwd() + f"\{fname}"
        self.opencst()

    def opencst(self):
        print("CST opening...")
        allpids = csti.running_design_environments()
        open = False
        for pid in allpids:
            self.de = csti.DesignEnvironment.connect(pid)
            # self.de.set_quiet_mode(True) # suppress message box
            print(f"Opening {self.full_path}...")
            try: self.prj = self.de.open_project(self.full_path)
            except: 
                print(f"Creating new project {self.full_path}")
                self.prj = self.de.new_mws()
                self.prj.save(self.full_path)
            open = True
            print(f"{self.full_path} open")
            break
        if not open:
            print("File path not found in current design environment...")
            print("Opening new design environment...")
            self.de = csti.DesignEnvironment.new()
            # self.de.set_quiet_mode(True) # suppress message box
            try: self.prj = self.de.open_project(self.full_path)
            except: 
                print(f"Creating new project {self.full_path}")
                self.prj = self.de.new_mws()
                self.prj.save(self.full_path)
            open = True
            print(f"{self.full_path} open")

    def read(self, result_item):
        results = cstr.ProjectFile(self.full_path, True) #bool: allow interactive
        try:
            res = results.get_3d().get_result_item(result_item)
            res = res.get_data()
        except:
            print("No result item.")
            available_files = results.get_3d().get_tree_items()
            closest_match = difflib.get_close_matches(result_item, available_files, n=1, cutoff=0.5)
            if closest_match: 
                result_item = closest_match[0] 
                print(f"Fetch '{result_item}' instead.")
            else: result_item = None
            res = results.get_3d().get_result_item(result_item)
            res = res.get_data()
        return res

    def save(self):
        self.prj.modeler.full_history_rebuild() 
        #update history, might discard changes if not added to history list
        self.prj.save()

    def close(self):
        self.de.close()

    def excute_vba(self,  command):
        command = "\n".join(command)
        vba = self.prj.schematic
        res = vba.execute_vba_code(command)
        return res

    def create_para(self,  para_name, para_value): #create or change are the same
        command = ['Sub Main', 'StoreDoubleParameter("%s", "%.4f")' % (para_name, para_value),
                'RebuildOnParametricChange(False, True)', 'End Sub']
        res = self.excute_vba (command)
        return command
    
    def create_shape(self, index, xmin, xmax, ymin, ymax, hc): #create or change are the same
        command = ['With Brick', '.Reset ', f'.Name "solid{index}" ', 
                   '.Component "component2" ', f'.Material "material{index}" ', 
                   f'.Xrange "{xmin}", "{xmax}" ', f'.Yrange "{ymin}", "{ymax}" ', 
                   f'.Zrange "0", "{hc}" ', '.Create', 'End With']
        return command
        # command = "\n".join(command)
        # self.prj.modeler.add_to_history(f"solid{index}",command)
    
    def create_cond_material(self, index, sigma, type="Lossy metal"): #create or change are the same
        command = ['With Material', '.Reset ', f'.Name "material{index}"', 
                #    '.Folder ""', '.Rho "8930"', '.ThermalType "Normal"', 
                #    '.ThermalConductivity "401"', '.SpecificHeat "390", "J/K/kg"', 
                #    '.DynamicViscosity "0"', '.UseEmissivity "True"', '.Emissivity "0"', 
                #    '.MetabolicRate "0.0"', '.VoxelConvection "0.0"', 
                #    '.BloodFlow "0"', '.MechanicsType "Isotropic"', 
                #    '.YoungsModulus "120"', '.PoissonsRatio "0.33"', 
                #    '.ThermalExpansionRate "17"', '.IntrinsicCarrierDensity "0"', 
                   '.FrqType "all"', f'.Type "{type}"', 
                   '.MaterialUnit "Frequency", "GHz"', '.MaterialUnit "Geometry", "mm"', 
                   '.MaterialUnit "Time", "ns"', '.MaterialUnit "Temperature", "Celsius"', 
                   '.Mu "1"', f'.Sigma "{sigma}"', 
                   '.LossyMetalSIRoughness "0.0"', '.ReferenceCoordSystem "Global"', 
                   '.CoordSystemType "Cartesian"', '.NLAnisotropy "False"', 
                   '.NLAStackingFactor "1"', '.NLADirectionX "1"', '.NLADirectionY "0"', 
                   '.NLADirectionZ "0"', '.Colour "0", "1", "1" ', '.Wireframe "False" ', 
                   '.Reflection "False" ', '.Allowoutline "True" ', 
                   '.Transparentoutline "False" ', '.Transparency "0" ', 
                   '.Create', 'End With']
        return command
        # command = "\n".join(command)
        # self.prj.modeler.add_to_history(f"material{index}",command)

    def set_frequency_solver(self):
        command = ['Sub Main', 'ChangeSolverType "HF Frequency Domain"', 
                   'Solver.FrequencyRange "1", "3"', 'End Sub']
        self.excute_vba(command)
        print("Frequency solver set")

    def set_time_solver(self):
        command = ['ChangeSolverType "HF Time Domain"', 
                   'Solver.FrequencyRange "1", "3"']
        command = "\n".join(command)
        self.prj.modeler.add_to_history("time_solver_and_freq_range",command)
        self.save()
        print("Time solver set")

    def start_simulate(self, plane_wave_excitation=False):
        print("Solving...")
        try: # problems occur with extreme conditions
            if plane_wave_excitation:
                command = ['Sub Main', 'With Solver', 
                '.StimulationPort "Plane wave"', 'End With', 'End Sub']
                self.excute_vba(command)
                print("Plane wave excitation = True")
            # one actually should not do try-except otherwise severe bug may NOT be detected
            model = self.prj.modeler
            model.run_solver()
        except Exception as e: pass
        print("Solved")
    
    def set_plane_wave(self):  # doesn't update history, disappear after save but remain after simulation
        command = ['Sub Main', 'With PlaneWave', '.Reset ', 
                   '.Normal "0", "0", "-1" ', '.EVector "1", "0", "0" ', 
                   '.Polarization "Linear" ', '.ReferenceFrequency "2" ', 
                   '.PhaseDifference "-90.0" ', '.CircularDirection "Left" ', 
                   '.AxialRatio "0.0" ', '.SetUserDecouplingPlane "False" ', 
                   '.Store', 'End With', 'End Sub']
        res = self.excute_vba(command)
        return res
    
    def set_excitation(self, filePath): # doesn't update history, disappear after save but remain after simulation. 
        # set .UseCopyOnly to false otherwise CST read cache
        command = ['Sub Main', 'With TimeSignal ', '.Reset ', 
                   '.Name "signal1" ', '.SignalType "Import" ', 
                   '.ProblemType "High Frequency" ', 
                   f'.FileName "{filePath}" ', 
                   '.Id "1"', '.UseCopyOnly "false" ', '.Periodic "False" ', 
                   '.Create ', '.ExcitationSignalAsReference "signal1", "High Frequency"',
                   'End With', 'End Sub']
        res = self.excute_vba(command)
        return res
    
    def delete_plane_wave(self):
        command = ['Sub Main', 'PlaneWave.Delete', 'End Sub']
        res = self.excute_vba(command)
        return res
    
    def delete_signal1(self):
        command = ['Sub Main', 'With TimeSignal', 
     '.Delete "signal1", "High Frequency" ', 'End With', 'End Sub']
        res = self.excute_vba(command)
        return res
    
    def set_port(self, point1, point2): # Not a robust piece of code, but anyway
        command = ['Sub Main', 'Pick.PickEdgeFromId "component1:feed", "1", "1"', 
                   'Pick.PickEdgeFromId "component1:coaxouter", "1", "1"', 
                   'With DiscreteFacePort ', '.Reset ', '.PortNumber "1" ', 
                   '.Type "SParameter"', '.Label ""', '.Folder ""', '.Impedance "50.0"', 
                   '.VoltageAmplitude "1.0"', '.CurrentAmplitude "1.0"', '.Monitor "True"', 
                   '.CenterEdge "True"', f'.SetP1 "True", "{point1[0]}", "{point1[1]}", "{point1[2]}"', 
                   f'.SetP2 "True", "{point2[0]}", "{point2[1]}", "{point2[2]}"', '.LocalCoordinates "False"', 
                   '.InvertDirection "False"', '.UseProjection "False"', 
                   '.ReverseProjection "False"', '.FaceType "Linear"', '.Create ', 
                   'End With', 'End Sub']
        res = self.excute_vba(command)
        return res
    
    def delete_port(self):
        command = ['Sub Main', 'Port.Delete "1"', 'End Sub']
        res = self.excute_vba(command)
        return res
    
    def export_E_field(self, outputPath, resultPath, time_end, time_step, d_step):
        total_samples = int(time_end/time_step)
        command = ['Sub Main',
        'SelectTreeItem  ("%s")' % resultPath, 
        'With ASCIIExport', '.Reset',
        f'.FileName ("{outputPath}")',
        f'.SetSampleRange(0, {total_samples})',
        '.Mode ("FixedWidth")', f'.Step ({d_step})',
        '.Execute', 'End With', 'End Sub']
        res = self.excute_vba(command)
        return res
    
    def export_power(self, outputPath, resultPath, time_end, time_step):
        total_samples = int(time_end/time_step)
        command = ['Sub Main',
        f'SelectTreeItem  ("{resultPath}")', 
        'With ASCIIExport', '.Reset',
        f'.FileName ("{outputPath}")',
        f'.SetSampleRange(0, {total_samples})',
        '.StepX (4)', '.StepY (4)',
        '.Execute', 'End With', 'End Sub']
        res = self.excute_vba(command)
        return res
    
    def delete_results(self):
        command = ['Sub Main',
        'DeleteResults', 'End Sub']
        res = self.excute_vba(command)
        return res
    
    def xz_symmetric_boundary(self): # don't know how to nonmanually delete though
        command = ['With Boundary', '.Xmin "expanded open"', 
                   '.Xmax "expanded open"', '.Ymin "expanded open"', '.Ymax "expanded open"', 
                   '.Zmin "expanded open"', '.Zmax "expanded open"', '.Xsymmetry "none"', 
                   '.Ysymmetry "magnetic"', '.Zsymmetry "none"', '.ApplyInAllDirections "False"', 
                   '.OpenAddSpaceFactor "0.5"', 'End With']
        command = "\n".join(command)
        self.prj.modeler.add_to_history("symmetric_boundary",command)
        self.save()
        print("Symmetric boundary set")

class Controller(CSTInterface):
    def __init__(self, fname):
        super().__init__(fname)
        self.Lg = LG
        self.Wg = WG
        self.hc = HC
        self.hs = HS
        self.feedx = FEEDX
        self.feedy = FEEDY
        self.Ld = L
        self.Wd = W
        self.d = D
        self.time_step = TSTEP
        self.time_end = TEND
        point1 = (self.feedx+self.hs/2-0.1, self.feedy, -5-self.hc-self.hs)
        point2 = (self.feedx+self.hs, self.feedy, -5-self.hc-self.hs)
        self.port = (point1, point2)


    # initialize ground, substrate, feed, and port
    def set_base(self):
        print("Setting base...")
        # Create ground, substrate, feed, and port
        ground = ['Component.New "component1"', 'Component.New "component2"',
                   'With Brick', '.Reset ', 
                   '.Name "ground" ', '.Component "component1" ', 
                   '.Material "Copper (annealed)" ', f'.Xrange "{-self.Lg/2}", "{self.Lg/2}" ', 
                   f'.Yrange "{-self.Wg/2}", "{self.Wg/2}" ', f'.Zrange "{-self.hc-self.hs}", "{-self.hs}" ', '.Create', 'End With']
        substrate = ['With Material', '.Reset', '.Name "FR-4 (loss free)"', 
                   '.Folder ""', '.FrqType "all"', '.Type "Normal"', 
                   '.SetMaterialUnit "GHz", "mm"', '.Epsilon "4.3"', '.Mu "1.0"', 
                   '.Kappa "0.0"', '.KappaM "0.0"', 
                   '.TanDM "0.0"', '.TanDMFreq "0.0"', '.TanDMGiven "False"', 
                   '.TanDMModel "ConstKappa"', '.DispModelEps "None"', 
                   '.DispModelMu "None"', '.DispersiveFittingSchemeEps "General 1st"', 
                   '.DispersiveFittingSchemeMu "General 1st"', 
                   '.UseGeneralDispersionEps "False"', '.UseGeneralDispersionMu "False"', 
                   '.Rho "0.0"', '.ThermalType "Normal"', '.ThermalConductivity "0.3"', 
                   '.SetActiveMaterial "all"', '.Colour "0.94", "0.82", "0.76"', 
                   '.Wireframe "False"', '.Transparency "0"', '.Create', 'End With',
                   'With Brick', '.Reset ', '.Name "substrate" ', 
                   '.Component "component1" ', '.Material "FR-4 (loss free)" ', 
                   f'.Xrange "{-self.Lg/2}", "{self.Lg/2}" ', f'.Yrange "{-self.Wg/2}", "{self.Wg/2}" ', 
                   f'.Zrange "{-self.hs}", "0" ', '.Create', 'End With ']
        ground_sub = ['With Cylinder ', '.Reset ', '.Name "sub" ', '.Component "component1" ', 
                   '.Material "Copper (annealed)" ', f'.OuterRadius "{self.hs}" ', 
                   '.InnerRadius "0.0" ', '.Axis "z" ', f'.Zrange "{-self.hc-self.hs}", "{-self.hs}" ', 
                   f'.Xcenter "{self.feedx}" ', f'.Ycenter "{self.feedy}" ', '.Segments "0" ', '.Create ', 
                   'End With', 'Solid.Subtract "component1:ground", "component1:sub"']
        substrate_sub = ['With Cylinder ', '.Reset ', '.Name "feedsub" ', 
                   '.Component "component1" ', '.Material "FR-4 (loss free)" ', 
                   f'.OuterRadius "{self.hs/2-0.1}" ', '.InnerRadius "0.0" ', '.Axis "z" ', 
                   f'.Zrange "{-self.hs}", "0" ', f'.Xcenter "{self.feedx}" ', f'.Ycenter "{self.feedy}" ', 
                   '.Segments "0" ', '.Create ', 'End With', 
                   'Solid.Subtract "component1:substrate", "component1:feedsub"'] 
        feed = ['With Cylinder ', '.Reset ', '.Name "feed" ', '.Component "component1" ', 
                   '.Material "PEC" ', f'.OuterRadius "{self.hs/2-0.1}" ', '.InnerRadius "0.0" ', 
                   '.Axis "z" ', f'.Zrange "{-5-self.hc-self.hs}", "{self.hc}" ', f'.Xcenter "{self.feedx}" ', 
                   f'.Ycenter "{self.feedy}" ', '.Segments "0" ', '.Create ', 'End With']
        coax = ['With Cylinder ', '.Reset ', '.Name "coax" ', '.Component "component1" ', 
                   '.Material "Vacuum" ', f'.OuterRadius "{self.hs-0.01}" ', f'.InnerRadius "{self.hs/2-0.1}" ', 
                   '.Axis "z" ', f'.Zrange "{-5-self.hc-self.hs}", "{-self.hc-self.hs}" ', f'.Xcenter "{self.feedx}" ', 
                   f'.Ycenter "{self.feedy}" ', '.Segments "0" ', '.Create ', 'End With', 
                   'With Cylinder ', '.Reset ', '.Name "coaxouter" ', 
                   '.Component "component1" ', '.Material "PEC" ', f'.OuterRadius "{self.hs}" ', 
                   f'.InnerRadius "{self.hs-0.01}" ', '.Axis "z" ', f'.Zrange "{-5-self.hc-self.hs}", "{-self.hc-self.hs}" ', 
                   f'.Xcenter "{self.feedx}" ', f'.Ycenter "{self.feedy}" ', '.Segments "0" ', '.Create ', 
                   'End With']
        command = ground + substrate + ground_sub + substrate_sub + feed + coax
        command = "\n".join(command)
        self.prj.modeler.add_to_history("initialize",command)
        self.save()
        print("Base set")
    
    def set_monitor(self):
        print("Setting monitor...")
        margin = (self.Ld - self.d)/2
        # Set monitor to read E field on domain
        EonPatch = ['With Monitor ', '.Reset ', '.Name "E_field_on_patch" ', 
                   '.Dimension "Volume" ', '.Domain "Time" ', '.FieldType "Efield" ', 
                   '.Tstart "0" ', f'.Tstep "{self.time_step}" ', f'.Tend "{self.time_end}" ', '.UseTend "True" ', 
                   '.UseSubvolume "True" ', '.Coordinates "Free" ', 
                   f'.SetSubvolume "0", "0", "0", "0", "{-5-self.hc-self.hs}", "{self.hc}" ', 
                   f'.SetSubvolumeOffset "{margin}", "{margin}", "{margin}", "{margin}", "{margin}", "{margin}" ', 
                   '.SetSubvolumeInflateWithOffset "True" ', '.PlaneNormal "z" ', 
                   f'.PlanePosition "{self.hc}" ', '.Create ', 'End With']
        command = EonPatch
        command = "\n".join(command)
        self.prj.modeler.add_to_history("set monitor",command)
        self.save()
        print("Monitor set")

    def set_domain(self): 
        print("Setting domain...")
        # Initialize domain with uniform conductivity
        nx, ny = (int(self.Ld//self.d), int(self.Wd//self.d))
        cond = np.zeros(nx*ny)
        print(f"{nx*ny} pixels in total...")
        # Define materials first
        self.update_distribution(cond)
        command = []
        # Define shape and index based on materials
        for index, sigma in enumerate(cond): 
            midpoint = (self.Ld/2, self.Wd/2)
            xi = index%nx
            yi = index//nx
            xmin = xi*self.d-midpoint[0]
            xmax = xmin+self.d
            ymin = yi*self.d-midpoint[1]
            ymax = ymin+self.d
            command += self.create_shape(index, xmin, xmax, ymin, ymax, self.hc)
        command = "\n".join(command)
        self.prj.modeler.add_to_history("domain",command)
        self.save()
        print("Domain set")

    def update_distribution(self, binary_sequence):
        print("Material distribution updating...")
        command_material = []
        for index, sigma in enumerate(binary_sequence):
            if sigma == 1: command_material += self.create_PEC_material(index)
            elif sigma == 0: command_material += self.create_air_material(index)
            else: print("undefined material")
        command_material = "\n".join(command_material)
        self.prj.modeler.add_to_history("material update",command_material)
        print("Material distribution updated")

    def create_PEC_material(self, index): #create or change are the same
        command = ['With Material', '.Reset ', f'.Name "material{index}"', 
                   '.FrqType "all"', '.Type "PEC"', 
                   '.MaterialUnit "Frequency", "GHz"', '.MaterialUnit "Geometry", "mm"', 
                   '.MaterialUnit "Time", "ns"', '.MaterialUnit "Temperature", "Celsius"',  
                   '.Create', 'End With']
        return command
    
    def create_air_material(self, index): #create or change are the same
        command = ['With Material', '.Reset ', f'.Name "material{index}"',  
                   '.FrqType "all"', '.Type "Normal"', 
                   '.MaterialUnit "Frequency", "GHz"', '.MaterialUnit "Geometry", "mm"', 
                   '.MaterialUnit "Time", "ns"', '.MaterialUnit "Temperature", "Celsius"', 
                   '.Epsilon "1"', '.Mu "1"', '.Create', 'End With']
        return command
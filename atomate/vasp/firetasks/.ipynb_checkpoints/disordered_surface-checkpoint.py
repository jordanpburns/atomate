from pymatgen import Structure, Lattice
from atomate.vasp.fireworks.core import OptimizeFW, StaticFW
from fireworks import LaunchPad, Firework, Workflow, FiretaskBase, FWAction, explicit_serialize
import glob
from datetime import datetime
from pytz import timezone
from fireworks import LaunchPad
from pymatgen.io.vasp.outputs import Outcar
from pymatgen.io.vasp.inputs import Poscar

@explicit_serialize
class CreateSlab(FiretaskBase):
    """
    Create a slab from an optimized structure and create slab and surface energy calculation fireworks. 

    Required params:
    vacuum: Angstroms of vacuum to add
    num_layers: number of layers of bulk structure to use
    surf_layers_to_relax: the number of surface layers to relax. The rest of the structure will
    be fixed with selective dynamics to simulate bulk
    atomic_thickness: how many atomic layers make up the bulk structure
    """

    #required_params = ["vacuum", "num_layers", "surf_layers_to_relax", "atomic_thickness"]
    required_params = []
    optional_params = []

    def run_task(self, fw_spec):

        #create structure from CONTCAR
        struct = Poscar.from_file('CONTCAR').structure

        #repeat layers of bulk to create supercell
        struct.make_supercell([1,1, self.get("num_layers", 2)])

        #add vacuum to create slab
        struct = add_vacuum(struct, self.get("vacuum", 15))

        #add selective dynamics
        selective_dynamics = []
        
        min_bulk = self.get("surf_layers_to_relax",3)/(self.get("atomic_thickness")*self.get("num_layers",2)) * max([site.z for site in struct.sites])
        
        max_bulk = (self.get("atomic_thickness")*self.get("num_layers",2) - self.get("surf_layers_to_relax",3))/(self.get("atomic_thickness")*self.get("num_layers",2)) * max([site.z for site in struct.sites])
                     
        for site in struct.sites:
            if site.z > min_bulk and site.z <= max_bulk:
                selective_dynamics.append([False, False, False])
            else:
                selective_dynamics.append([True, True, True])
        struct.add_site_property("selective_dynamics", selective_dynamics)
        
        #create optimize and static fireworks using the newly created slab
        slab_optimize = OptimizeFW(struct, name = name + '_slab_optimization' + time, vasp_cmd=">>vasp_cmd<<", db_file=">>db_file<<", parents = [bulk_optimize])

        slab_static = StaticFW(struct, name = name + '_slab_static_' + time, parents = [slab_optimize], prev_calc_loc=True, vasp_cmd=">>vasp_cmd<<", db_file=">>db_file<<")
        slab_static.tasks.append(Slab_energy_and_SA())        
                     
        surface_energy_calc_fw = Firework(Surface_energy_calc(), parents = [bulk_static, slab_static])
                     
        return FWAction(additions = [slab_optimize, slab_static, surface_energy_calc_fw])

    def add_vacuum(struct, c_to_add):
        """This method adds vacuum to a Structure. It assumes the structure is on the bottom
        and exisiting vacuum is on top (and a c-axis aligned with the slab_normal)
        """

        new_lattice = struct.lattice.matrix.copy()
        new_lattice[2, 2] += c_to_add
        return Structure(Lattice(new_lattice), struct.species, struct.cart_coords, coords_are_cartesian=True, to_unit_cell=True, site_properties=struct.site_properties)

@explicit_serialize
class Surface_energy_calc(FiretaskBase):
    """This firetask gets the energies from the static calculations of the bulk and slab calculations as well as the surface area and then calculates the surface energy"""
    
    #required_params = ["bulk_energy", "slab_energy", "surface_area", "num_layers"]
    required_params = []
    optional_params = []
    
    def run_task(self, fw_spec):
        bulk_E = self.get("bulk_energy")
        slab_E = self.get("slab_energy")
        surface_area = self.get("surface_area")
        n = self.get("num_layers") 
        surface_energy = (slab_E - n * bulk_E) / (2 * surface_area)
        return FWAction(mod_spec=[{'_push': {'surface_energy': surface_energy}}])
            
@explicit_serialize
class Bulk_energy(FiretaskBase):
    """This firetask gets the energy of the bulk static calculation and adds it to the child firework spec"""
            
    required_params = []
    optional_params = []
    
    def run_task(self, fw_spec):
            
        OUTCAR = Outcar("OUTCAR")
        energy = OUTCAR.final_energy
        return FWAction(mod_spec=[{'_push': {'bulk_energy': energy}}])

@explicit_serialize
class Slab_energy_and_SA(FiretaskBase):
    """This firetask gets the energy of the slab static calculation as well as the surface area and adds it to the child firework spec"""
            
    required_params = []
    optional_params = []
    
    def run_task(self, fw_spec):
            
        OUTCAR = Outcar("OUTCAR")
        energy = OUTCAR.final_energy
        
        POSCAR = Poscar.from_file("CONTCAR")
        a = POSCAR.structure.lattice.a
        b = POSCAR.structure.lattice.b
        surface_area = a * b
        
        
        return FWAction(mod_spec=[{'_push': {'slab_energy': energy, 'surface_area':surface_area}}])
import java.util.*;
import java.util.List;
import java.util.function.BiFunction;
import java.util.regex.Pattern;
import java.util.regex.Matcher;


/**
 * Simple two-coordinates container.
 */
class Point{
    int x;
    int y;
    public Point(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public void moveTo(int x, int y) {
        this.x = x;
        this.y = y;
    }

    /**
     * Static method producing Point object from output
     * @param pointRepr string to be converted
     * @return Point object
     * @throws MalformedInputException the string input is ill-formed and not corresponds to the specification.
     */
    public static Point fromString(String pointRepr) throws MalformedInputException {
        try {
            Pattern p = Pattern.compile("(\\d+),(\\d+)");
            Matcher m = p.matcher(pointRepr);
            if (m.find()) {
                int column = Integer.parseInt(m.group(1));
                int row = Integer.parseInt(m.group(2));
                if(!(column >= 0 && column < 9 && row >= 0 && row < 9)) throw new Exception();
                return new Point(row, column);
            }
            return null;
        } catch(Exception ex) {
            throw new MalformedInputException("Malformed cell format. Please input cells in format [0-8,0-8]");
        }
    }

        public String toString() {
            return String.format("[%d,%d]", this.y, this.x);
        }
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Point point = (Point) o;
        return (x == point.x) && (y == point.y);
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y);
    }
}

/**
 * Base class for agents on lattice
 */
class CellularAgent {
    public Point pos;
    public int mooreN;
    public int initialMooreN;

    public CellularAgent(int x, int y, int mooreN) {
        this(new Point(x,y), mooreN);
    }

    public CellularAgent(Point pos, int mooreN) {
        this.pos = pos;
        this.mooreN = mooreN;
        this.initialMooreN = mooreN;
    }

    /**
     * Moves agent to new position.
     * @param newPos new position of the agent.
     */
    public void moveTo(Point newPos) {
        pos = newPos;
    }
    public void moveTo(int x, int y){
        pos.moveTo(x, y);
    }


    /**
     * Returns a perception zone of agent, i.e. a set of cells that agent can perceive at its current position
     *
     * @return A list of perceived cells.
     */
    ArrayList<Point> getPerceptionZone() {
        ArrayList<Point> perceptionZone = new ArrayList<>(9);
        if(mooreN >= 0) {
            for(int i = -mooreN; i <= mooreN; i++) {
               for(int j = -mooreN; j <= mooreN; j++) {
                   Point p = new Point(pos.x+i, pos.y+j);
                   if(CellularEnvironment.getInstance().isValidPoint(p))
                       perceptionZone.add(p);
               }
            }
        }
        else { // perception scenario 2
            assert false;
        }
        return perceptionZone;
    }


    /**
     * Checks if at least one point of List pts can be perceived by the agent.
     * @param pts Points to consider.
     * @return true if at least point of pts can be perceived; false otherwise.
     */
    boolean perceivesOneOrMany(List<Point> pts) {
        for(Point p : pts) {
            if(perceives(p)) return true;
        }
        return false;
    }

    /**
     * Checks if a given point is in range of agent's perception.
     * @param p asked point
     * @return true if agent can perceive objects in this point; false otherwise;
     */
    boolean perceives(Point p) {
        if (mooreN >= 0) {
            int ix = this.pos.x;
            int iy = this.pos.y;
            int r = this.mooreN;

            if ((ix + r >= p.x) && (ix - r <= p.x) && (iy + r >= p.y) && (iy - r <= p.y)) {
                return true;
            }
        }
        else {
            return this.getPerceptionZone().contains(p);
        }
        return false;
    }


}

/**
 * Class for inspector agents.
 */
class InspectorAgent extends CellularAgent {
    public InspectorAgent(Point pos, int mooreN) {
        super(pos, mooreN);
    }
    public InspectorAgent clone() {
        return new InspectorAgent(super.pos, super.mooreN);
    }
}

/**
 * Class for actor agents (namely, Harry)
 */
class ActorAgent extends CellularAgent {
    static enum Algorithms{
        A_STAR,
        BACKTRACK
    }

    enum Goals{
        BOOK,
        CLOAK,
        EXIT
    }
    Algorithms currentAlgo;
    private Point initialPos;
    private Point exitLocation, cloakLocation, bookLocation;
    boolean[][] knownDangers;
    public ArrayDeque<Point> bookPath, exitPath;
    public ArrayDeque<Point> path;
    private boolean gameOver;
    private CellularEnvironment env;



    public ActorAgent(Point pos, int mooreN, CellularEnvironment env) {
        super(pos, mooreN);
        this.env = CellularEnvironment.getInstance();
        initialPos = super.pos;
        this.bookPath = new ArrayDeque<>();
        this.exitPath = new ArrayDeque<>();
        this.exitLocation = env.getExitLocation();
    }

    /**
     * Overrides the superclass' method; adds the implementation of scenario 2 with non-Moore perception zone.
     * @return ArrayList of perceived cells.
     */
    @Override
    ArrayList<Point> getPerceptionZone() {
        ArrayList<Point> perceptionZone = new ArrayList<>(9);
        if(mooreN >= 0) {
            for(int i = -mooreN; i <= mooreN; i++) {
                for(int j = -mooreN; j <= mooreN; j++) {
                    Point p = new Point(pos.x+i, pos.y+j);
                    if(env.isValidPoint(p))
                        perceptionZone.add(p);
                }
            }
        }
        else { // perception scenario 2
            int[] range = new int[]{-2, 2};
            for(int i : range) {
                for(int j = -1; j <= 1; j++){
                    Point p = new Point(pos.x+i, pos.y+j);
                    if(env.isValidPoint(p))
                        perceptionZone.add(p);
                }
            }
            for(int i = -1; i <= 1; i++) {
                for(int j : range) {
                    Point p = new Point(pos.x+i, pos.y+j);
                    if(env.isValidPoint(p))
                        perceptionZone.add(p);
                }
            }
            perceptionZone.add(new Point(pos.x, pos.y));
        }
        return perceptionZone;
        }


    /**
     * Search dispatcher. Specifies the algorithm a goal to be searched.
     * @param startPos  position where the search begins
     * @param goal      object to be found
     * @param path      deque that will contain the result of the search (if successful)
     * @param algo      applied search algorithm.
     * @return          true if the goal is found; false otherwise.
     */
    private boolean search(Point startPos, Goals goal, Deque<Point> path, Algorithms algo) {
        BiFunction<Point, Point, Integer> heuristic = ActorAgent::zeroHeuristic;
        Point goalPoint = null;
        switch(goal){
            case CLOAK:
                if(this.cloakLocation != null){
                    heuristic = ActorAgent::euclideanHeuristic;
                    goalPoint = this.cloakLocation;
                }
                break;
            case BOOK:
                if(this.bookLocation != null) {
                    heuristic = ActorAgent::euclideanHeuristic;
                    goalPoint = this.cloakLocation;
                }
                break;
            case EXIT:
                heuristic = ActorAgent::euclideanHeuristic;
                goalPoint = this.exitLocation;
        }
        if(algo == Algorithms.BACKTRACK)
            return searchBacktrack(startPos, goal, path);
        else if(algo == Algorithms.A_STAR)
            return searchAStar(startPos, goal, goalPoint, path, heuristic);
        else
            assert(false);
        return false;
    }

    /**
     * Method implementing the book-finding problem stated in the assignment available for clients.
     * Searches for tour that Harry can take to pick the book and reach the exit without being caught.
     *
     * @return      true if the solution is found; false otherwise.
     */
    public boolean find() {
        // remove everything that could be found on the previous run
        this.bookLocation = null;
        this.cloakLocation = null;
        this.bookPath = null;
        this.exitPath = null;
        this.path = null;
        // the problem: finding the cloak drastically reduces path constraints for agents.a
        // that yields longer paths than in case the agent needs to avoid larger Moore neighbourhoods of inspectors'
        // perception. So, it is good idea to try achieving the goal without using the cloak.

        // Searching the cloak alone is not going to affect the solution path since we have the same tree to search
        // with just different halting condition.

        // option A: try to find the path directly without taking the cloak (if possible).
        ArrayDeque<Point> bookPathA = new ArrayDeque<>();
        ArrayDeque<Point> exitPathA = null;
        boolean foundBookA = search(this.initialPos, Goals.BOOK,bookPathA, currentAlgo),
                foundExitA = false;
        if (foundBookA) {
            exitPathA = new ArrayDeque<>();
            foundExitA = search(bookPathA.peekLast(), Goals.EXIT, exitPathA, currentAlgo);
        }
        boolean foundA = foundBookA && foundExitA;
        ArrayDeque<Point> pathA = null;
        if(foundA)
        {
            pathA = bookPathA.clone();
            pathA.pollLast();
            Iterator<Point> it = exitPathA.iterator();
            while(it.hasNext()) {
                pathA.addLast(it.next());
            }
        }

        // option B: try to get to the book without the cloak, but get the cloak on the way back (if possible)
        // the first part is identical to the one of option A
        ArrayDeque<Point> bookPathB = bookPathA.clone(), exitPathB = null;
        boolean foundBookB = foundBookA, foundExitB = false;
        if (foundBookA) {
            exitPathB = new ArrayDeque<>();
            ArrayDeque<Point> cloakPathB = new ArrayDeque<>();
            if (search(bookPathB.peekLast(), Goals.CLOAK, cloakPathB, currentAlgo)) {
                this.knownDangers = null; // since most cells are not threats to us anymore, reset the knowledge of dangerous cells
                env.setHasCloak(true);
                foundExitB = search(cloakPathB.peekLast(), Goals.EXIT, exitPathB, currentAlgo);
                while(!cloakPathB.isEmpty()){
                    exitPathB.addFirst(cloakPathB.pollLast());
                }
            }
            env.setHasCloak(false);
        }
        boolean foundB = foundBookB && foundExitB;
        ArrayDeque<Point> pathB = null;
        if(foundB)
        {
            pathB = bookPathB.clone();
            pathB.pollLast();
            Iterator<Point> it = exitPathB.iterator();
            while(it.hasNext()) {
                pathB.addLast(it.next());
            }
        }

        // option C: pick the cloak first (and use it until reaching exit)
        boolean foundBookC = false, foundExitC = false;
        ArrayDeque<Point> bookPathC = new ArrayDeque<>();
        ArrayDeque<Point> exitPathC = new ArrayDeque<>();
        ArrayDeque<Point> cloakPathC = new ArrayDeque<>();

        if (search(initialPos, Goals.CLOAK, cloakPathC, currentAlgo)) {
            this.knownDangers = null; // since most of the cells are not threats to us anymore, reset the knowledge of dangerous cells
            env.setHasCloak(true);
            foundBookC = search(cloakPathC.peekLast(), Goals.BOOK, bookPathC, currentAlgo);
            if (foundBookC) {
                while(!cloakPathC.isEmpty()){
                    bookPathC.addFirst(cloakPathC.pollLast());
                }
                foundExitC = search(bookPathC.peekLast(), Goals.EXIT, exitPathC, currentAlgo);
            }
            env.setHasCloak(false);
        }
        boolean foundC = foundBookC && foundExitC;
        ArrayDeque<Point> pathC = null;
        if(foundC)
        {
            pathC = bookPathC.clone();
            pathC.pollLast();
            Iterator<Point> it = exitPathC.iterator();
            while(it.hasNext()) {
                pathC.addLast(it.next());
            }
        }

        if(foundA || foundB || foundC) {
            ArrayList<ArrayDeque<Point>> paths = new ArrayList<>(3);
            if(pathA != null) paths.add(pathA); if(pathB != null) paths.add(pathB); if(pathC != null) paths.add(pathC);
            paths.sort(new PathsComparator());
            this.path = paths.get(0);
            if(path.equals(pathA)) {
                bookPath = bookPathA;
                exitPath = exitPathA;
            }
            else if(path.equals(pathB)){
                bookPath = bookPathB;
                exitPath = exitPathB;
            }
            else if(path.equals(pathC)){
                bookPath = bookPathC;
                exitPath = exitPathC;
            }
            else {
                assert false;
            }
            return true;
        }
        return false;
    }

    /**
     * Wrapper for A* search.
     * @param startPos      point where the search begins.
     * @param goal          object to be found.
     * @param goalPoint     Point where the goal object is located (if known). Can be null if the location of the goal is unknown.
     * @param path          A sequence of points that represents the solution.
     * @param heuristic     Heuristic function, dispatched in the search() method of the same class.
     * @return              True if the goal is found; false otherwise.
     */
    private boolean searchAStar(Point startPos, Goals goal, Point goalPoint, Deque<Point> path, BiFunction<Point, Point, Integer> heuristic){
        if (knownDangers == null)
            knownDangers = new boolean[env.worldRows][env.worldColumns];
        return _searchAStar(startPos, goal, goalPoint, path, heuristic);
    }

    /**
     * Actual A* implementation.
     * @param startPos      point where the search begins.
     * @param goal          object to be found.
     * @param goalPoint     Point where the goal object is located (if known). Can be null if the location of the goal is unknown.
     * @param path          A sequence of points that represents the solution.
     * @param heuristic     Heuristic function, dispatched in the search() method of the same class.
     * @return              True if the goal is found; false otherwise.
     */
    private boolean _searchAStar(Point startPos, Goals goal, Point goalPoint, Deque<Point> path, BiFunction<Point, Point, Integer> heuristic) {
        State[][] states = new State[env.worldRows][env.worldColumns];
        PriorityQueue<State> openStates = new PriorityQueue<>(new StateCostComparator());
        HashSet<Point> closedStates = new HashSet<>();

        State state = new State(startPos, 0, null);
        state.fCost = state.gCost + heuristic.apply(state.point, goalPoint);

        states[startPos.x][startPos.y] = state;
        openStates.offer(state);
        State goalState = null;

        while(!openStates.isEmpty()) {
            State current = openStates.poll();
            this.moveTo(current.point);
            env.senseDangers(this);

            if(env.isCaughtInCell(pos)){
                return false;
            }

            if(env.isBookAt(pos)) {
                this.bookLocation = current.point;
            }

            if(env.isBookAt(pos)) {
                this.cloakLocation = current.point;
            }

            if(isGoalAt(pos, goal)) {
                goalState = current;
                break;
            }


            for (Point neighbor : env.neighboursOfCell(current.point)) {
                if(closedStates.contains(neighbor)) continue;
                State neighState = states[neighbor.x][neighbor.y];
                if(neighState == null) {
                    neighState = new State(neighbor, -1, current);
                    neighState.fCost = state.gCost + heuristic.apply(neighState.point, goalPoint);
                    states[neighbor.x][neighbor.y] = neighState;
                }
                int neighGCost = current.gCost + 1; // assign actual cost (current cell + unit transition cost)
                int neighFCost = evaluationFunction(neighbor, neighGCost, goalPoint, heuristic);
                if(neighState.gCost == -1 || neighGCost < neighState.gCost){
                    neighState.prev = current;
                    neighState.gCost = neighGCost;
                    neighState.fCost = neighFCost;
                    if (!openStates.contains(neighState))
                        openStates.offer(neighState);
                }
            }
            closedStates.add(current.point);
        }
        if (goalState == null) return false;
        while(goalState != null) {
            path.addFirst(goalState.point);
            goalState = goalState.prev;
        }
        return true;
    }

    /**
     * Class representing the state of a system. Used in A* search.
     */
    private class State {
        State prev;
        int gCost;
        int fCost;
        Point point;
        boolean hasCloak;
        public State(Point point, int gCost, State prev) {
            this.point = point;
            this.prev = prev;
            this.gCost = gCost;
        }
    }

    /**
     * Evaluation function for A* search.
     * @param cell          cell evaluation function will be computed for
     * @param gCost         cost of that cell
     * @param goalPoint     point we want to get to
     * @param heuristic     heuristic function that will be computed as a part of evaluation function
     * @return              evaluated cost of the given cell.
     */
    private int evaluationFunction(Point cell, int gCost, Point goalPoint, BiFunction<Point, Point, Integer> heuristic) {
        int eval;
        if(knownDangers[cell.x][cell.y])
            eval = Integer.MAX_VALUE-10;
        else
            eval = gCost + heuristic.apply(cell, goalPoint);
        return eval;
    }

    /**
     * Heuristic calculating the Euclidean distance between two points.
     * @param cell           first point
     * @param goalPoint      second point
     * @return               Euclidean distance between two points.
     */
    private static int euclideanHeuristic(Point cell, Point goalPoint) {
        int dist = (int) Math.sqrt((goalPoint.x - cell.x)*(goalPoint.x - cell.x) + (goalPoint.y - cell.y)*(goalPoint.y - cell.y));
        return dist;
    }

    /**
     * Zero heuristic for objects with unknown location.
     */
    private static int zeroHeuristic(Point cell, Point target) {
        return 0;
    }


    class StateCostComparator implements Comparator<State> {
        /**
         * Comparator for A* priority queue to work. Performs sorting by costs produced by evaluation function.
         * @param o1        first State object
         * @param o2        second State object.
         * @return          numerical value for Comparator in priority queue to work.
         */
        @Override
        public int compare(State o1, State o2) {
            return Integer.compare(o1.fCost, o2.fCost);
        }
    }

    /**
     * Comparator used in priority queue when we compare three solutions (start->book->exit, start->cloak->book->exit, etc.)
     * to find the deque containing the shortest path.
     */
    class PathsComparator implements Comparator<Deque<Point>> {
        @Override
        public int compare(Deque<Point> o1, Deque<Point> o2) {
            return Integer.compare(o1.size(), o2.size());
        }
    }

    /**
     * Performs the dispatching and asks the environment is the current cell contains the required goal.
     * @param cell  cell to be examined
     * @param goal  desired goal
     * @return      true if the cell contains the goal object; false otherwise.
     */
    private boolean isGoalAt(Point cell, Goals goal) {
        switch(goal){
            case BOOK:
                return env.isBookAt(cell);
            case CLOAK:
                return env.isCloakAt(cell);
            case EXIT:
                return cell.equals(this.exitLocation);
            default:
                assert(false);
                return false;
        }
    }


        /**
         * Wrapper for actual recursive backtrack search function.
         * @param cell      starting cell
         * @param goal      goal object to be found
         * @param path      will contain path solution in case of successful search.
         * @return          true if the goal is found; false otherwise
         */
        // The signature is notably simple compared to A*
        // since the backtracking is uninformed and doesn't require heuristic function or point location
        private boolean searchBacktrack (Point cell, Goals goal, Deque <Point> path){
            if(knownDangers == null)
                knownDangers = new boolean[env.worldRows][env.worldColumns];
            boolean[][] visited = new boolean[env.worldRows][env.worldColumns];
            gameOver = false;
            return _searchBacktrack(cell, goal, path,
                    visited);
        }



        /**
         * Performs a recursive search using backtracking algorithm.
         * @param cell      Initial cell.
         * @param goal      object to be found
         * @param path      will contain path in case of successful search
         * @param visited   keeps track of visited cells.
         * @return true if backtracking succeeded
         */
        private boolean _searchBacktrack(Point cell, Goals goal, Deque <Point> path, boolean[][] visited){
            this.moveTo(cell);
            visited[cell.x][cell.y] = true;
            env.senseDangers(this);

            if (env.isCaughtInCell(cell)) {
                knownDangers[pos.x][pos.y] = true;
                gameOver = true;    // by commenting out this we make the actor "immortal", which allows him to continue search even if it
                                    // stepped into inspector's zone.
                return false;
            }

            if (isGoalAt(cell, goal)) {
                path.addLast(cell);
                return true;
            }

            ArrayList<Point> neighbours =env.neighboursOfCell(cell);
            for(Point neighbor : neighbours) {
                if (!visited[neighbor.x][neighbor.y] && !knownDangers[neighbor.x][neighbor.y]) {
                    if (_searchBacktrack(neighbor, goal, path, visited)) {
                        path.addFirst(cell);
                        return true;
                    }
                    if(gameOver)
                        return false;
                }
            }
            return false;
        }

    }


/**
 * Singleton class describing the environment agents operate in.
 */
class CellularEnvironment {
    public final int worldColumns;
    public final int worldRows;
    private InspectorAgent[] inspectors;
    private Point bookLocation;
    private Point cloakLocation;
    private Point exitLocation;
    private boolean hasCloak = false;


    public InspectorAgent[] getInspectors() {
        return inspectors;
    }

    public Point getExitLocation() {
        return exitLocation;
    }

    public Point getBookLocation() {
        return bookLocation;
    }

    public Point getCloakLocation() {
        return cloakLocation;
    }

    private static CellularEnvironment instance;

    public static CellularEnvironment getInstance() {
        return instance;
    }

    public static CellularEnvironment getInstance(InspectorAgent[] inspectors,
                                                  Point bookLocation,
                                                  Point exitLocation,
                                                  Point cloakLocation,
                                                  int worldColumns, int worldRows) {
        instance = new CellularEnvironment(inspectors, bookLocation, exitLocation, cloakLocation, worldColumns, worldRows);
        return instance;
    }


    private CellularEnvironment(InspectorAgent[] inspectors,
                               Point bookLocation,
                               Point exitLocation,
                               Point cloakLocation,
                               int worldColumns, int worldRows) {
        this.inspectors = inspectors;
        this.bookLocation = bookLocation;
        this.cloakLocation = cloakLocation;
        this.exitLocation = exitLocation;
        this.worldColumns = worldColumns;
        this.worldRows = worldRows;
    }

    public boolean isCloakAt(Point cell) {
        return cell.equals(cloakLocation);
    }

    public boolean isBookAt(Point cell) {
        return cell.equals(bookLocation);
    }


    /**
     * Toggles the perception zones of inspectors depending on whether Harry has the cloak or not.
     * @param value
     */
    void setHasCloak(boolean value) {
        hasCloak = value;
        for (InspectorAgent inspector : inspectors)
            if(hasCloak)
                inspector.mooreN = 0;
            else
                inspector.mooreN = inspector.initialMooreN;
    }

    /**
     * Checks if the current cell is in perception zone of at least one of the agents (i.e, is the agent in that cell is caught)
     * @param cell      examined cell.
     * @return
     */
    boolean isCaughtInCell(Point cell) {
        for(InspectorAgent inspector : inspectors)
            if (inspector.perceives(cell))
               return true;
        return false;
    }

    /**
     * Gives the actor way to perceive all dangers in its perception zone (and memorize it if needed).
     * @param actor   perceiving actor.
     */
    void senseDangers(ActorAgent actor) {
        boolean result = false;
        for(Point cell : actor.getPerceptionZone()) {
            if(actor.knownDangers[cell.x][cell.y] == true) continue;
            for (InspectorAgent inspector : inspectors) {
                if (inspector.getPerceptionZone().contains(cell)) {
                    actor.knownDangers[cell.x][cell.y] = true;
                }
            }
        }
    }

    /**
     * Checks if the point is in range of the world.
     * @param p examined point.
     * @return   true if the point is valid; false otherwise.
     */
    boolean isValidPoint(Point p) {
       if(p.x >= 0 && p.x < worldRows && p.y >= 0 && p.y < worldColumns)
           return true;
       return false;
    }


    /**
     * Returns neighbours (adjacent cells) of given cell.
     * @param       cell examined cell.
     * @return      all cells adjacent to given one.
     */
    ArrayList<Point> neighboursOfCell(Point cell) {
        ArrayList<Point> neighbours = new ArrayList<>();
        for(int i = -1; i <= 1; ++i) {
            for(int j = -1; j <= 1; ++j) {
                if(j == 0 && i == 0) continue;
                if( cell.x + i >= 0 &&
                    cell.y + j >= 0 &&
                    cell.x + i < worldColumns &&
                    cell.y + j < worldRows ) {
                    neighbours.add(new Point(cell.x+i, cell.y+j));
                }
            }
        }
        return neighbours;
    }

    /**
     * @return deep copy of current object.
     */
    public CellularEnvironment clone(){
        InspectorAgent[] cloneInspectors = new InspectorAgent[inspectors.length];
        for(int i = 0; i < inspectors.length; i++){
            cloneInspectors[i] = inspectors[i].clone();
            assert(cloneInspectors[i] != inspectors[i]);
        }
        return new CellularEnvironment( cloneInspectors,
                bookLocation,
                exitLocation,
                cloakLocation,
        worldColumns, worldRows);
    }

}

/**
 * Exception raised when ill-formed input is encountered.
 */
class MalformedInputException extends Exception {
    public MalformedInputException(String message) {
        super(message);
    }
}

/**
 * Class used for keeping track of search algorithms' statistics.
 */
class Stats {
    int wins;
    ArrayList<Integer> steps; // arrays are required later to compute the variance
    ArrayList<Double> runtime;
    float avgSteps;
    double avgRuntime;
    double varianceRt, varianceSteps;
    public Stats(int total) {
        steps = new ArrayList<>(total);
        runtime = new ArrayList<>(total);
    }
}

/**
 * Main class with some driver code.
 */
public class Assignment1 {
    private static ActorAgent harry;
    private static CellularEnvironment cenv;
    private static Visualizer vis;
    private static long execTime;

    enum GenerationModes {
        MANUAL,
        SEMIAUTO,
        FULLAUTO
    }
    private static GenerationModes genMode = GenerationModes.MANUAL;
    private static int tests = 1;
    private static int scenario = -1;
    private static boolean showMaps = true;


    /**
     * Handles the commandline arguments configuring the program.
     * @param args      String array with arguments received from main
     */
    private static void handleArgs(String[] args) {
        try {
            for (String argument : args) {
                String cmd = argument.split("=")[0];
                String val = argument.split("=")[1];
                switch (cmd) {
                    case "generationmode":
                        switch (val) {
                            case "semiauto":
                                genMode = GenerationModes.SEMIAUTO;
                                break;
                            case "fullauto":
                                genMode = GenerationModes.FULLAUTO;
                                break;
                            case "manual":
                                genMode = GenerationModes.MANUAL;
                                break;
                            default:
                                throw new MalformedInputException("Generation mode not found. Available modes: {manual, semiauto, fullauto}");
                        }
                        break;
                    case "tests":
                        tests = Integer.parseInt(val);
                        break;
                    case "scenario":
                        scenario = Integer.parseInt(val);
                        if(scenario != 1 && scenario != 2)
                            throw new MalformedInputException("Scenario must be either 1 or 2, got: " + scenario);
                        break;
                    case "showmaps":
                        showMaps = Boolean.parseBoolean(val);
                        break;
                    default:
                        throw new MalformedInputException(String.format("Parameter %s not found", cmd));
                }
            }
            if (genMode != GenerationModes.FULLAUTO && scenario > 0) {
                throw new MalformedInputException("Scenario can be only chosen for fullauto generation mode");
            }
            else if(genMode == GenerationModes.FULLAUTO && scenario < 0) {
                throw new MalformedInputException("Please specify scenario={1,2} for fullauto generation mode");
            }
        } catch (IndexOutOfBoundsException iobEx) {
            System.out.println("Please input parameters in form parameter=value");
            System.exit(1);
        } catch(NumberFormatException nfeEx) {
            System.out.println("Please input tests= and scenario= parameter arguments in numerical format");

        } catch (MalformedInputException miEx) {
            System.out.printf("Argument parsing error: %s\n", miEx.getMessage());
            System.out.println("Usage: \n generationmode={manual, semiauto, fullauto} tests=<number of tests> scenario=<scenario number for fullauto mode> showmaps={true,false}\n" +
                    "\nprovides generation of test cases, either manual, semi-auto (scenario is put manually every test), " +
                    "or full-auto (scenario is chosen at start)" +
                    "\nno arguments is equivalent to generationmode=manual tests=1 showmaps=true");
            System.exit(1);
        }
    }

    private static Stats backTrackStats;
    private static Stats aStarStats;
    private static Scanner in;

    /**
     * Record the stat data in corresponding object in case of search success.
     * @param stat      Stats object to be written into
     * @param success   flag for successful search
     */
    private static void writeStats(Stats stat, boolean success){
        if(!success) return;
        stat.wins++;
        stat.steps.add(harry.path.size());
        stat.runtime.add((double) execTime / 1000);
        stat.avgRuntime += (double) execTime / 1000;
        stat.avgSteps += harry.path.size();
    }


    /**
     * Prints the statistics quantities (see the report)
     * @param stat        stats object storing statistics
     * @param algo        used algorithm
     */
    static void printStats(Stats stat, ActorAgent.Algorithms algo) {
        float winrate = (float)stat.wins/tests;
        stat.avgSteps /= stat.steps.size();
        stat.avgRuntime /= (stat.runtime.size());
        stat.varianceRt = 0; stat.varianceSteps = 0;
        assert(stat.steps.size() == stat.runtime.size());
        for(int i = 0; i < stat.steps.size(); i++) {
           double rt = stat.runtime.get(i);
           int steps = stat.steps.get(i);
           stat.varianceRt += (stat.avgRuntime - rt)*(stat.avgRuntime - rt);
           stat.varianceSteps += (stat.avgSteps - steps)*(stat.avgSteps - steps);
        }
        stat.varianceRt /= stat.runtime.size() - 1;
        stat.varianceSteps /= stat.steps.size() -1;
        System.out.printf("Statistics for search algorithm %s, perception scenario %d:\n", algo.toString(), scenario);
        System.out.printf("Tests: %2d\t\t\t\tWins: %2d\t\t\t\tLoses: %2d\t\t\t\tWin rate: %2.5f\n" +
                        "Avg. runtime (uS) (sample mean): %2.1f\t\t\tAvg. # of steps (sample mean): %2.1f\n",
                tests, stat.wins, tests-stat.wins, winrate,
                 stat.avgRuntime,stat.avgSteps);
        System.out.printf("Runtime sample variance: %2.1f\t\t\t\tSteps sample variance:  %2.1f\n", stat.varianceRt, stat.varianceSteps);

    }

    public static void main(String[] args) {
        handleArgs(args);
        // init stats containers
        backTrackStats = new Stats(tests);
        aStarStats = new Stats(tests);
        System.out.printf("####################################\nInput mode is %s, number of tests: %d\n####################################\n", genMode.toString(), tests);
        for(int i = 0; i < tests; i++) { // run for given number of tests
            if(scenario > 0 && genMode == GenerationModes.SEMIAUTO) scenario = -1; // reset scenario for semiauto mode
            System.out.printf("************************************\nTest %d/%d\n", i+1, tests);
            init();         // read the input if any and initialize structures
            boolean successBT = run(ActorAgent.Algorithms.BACKTRACK); // try backtracking search
            writeStats(backTrackStats, successBT);             // write stats in case of success
            visualize(successBT, showMaps);               // visualize the results (runtime, path, map, etc.)
            boolean successAS = run(ActorAgent.Algorithms.A_STAR);   // perform the same for A*
            writeStats(aStarStats, successAS);
            visualize(successAS, showMaps);
            System.out.printf("\n************************************\n");
        }
        printStats(backTrackStats, ActorAgent.Algorithms.BACKTRACK);
        System.out.println("=======================");
        printStats(aStarStats, ActorAgent.Algorithms.A_STAR);
        System.out.println("=======================");
        calculateT(backTrackStats, aStarStats);
        System.out.println("=======================");
    }


    /**
     * Calculates the t-value for statistics report.
     * @param stat1     first statistics container (for backtracking, for example)
     * @param stat2     second statistics container (for A*)
     */
    static void calculateT(Stats stat1, Stats stat2) {
        double s1 = stat1.varianceRt;
        double s2 = stat2.varianceRt;
        int N = Math.min(stat1.runtime.size()-1, stat2.runtime.size()-1);
        double s_x1x2 = Math.sqrt( (s1 + s2) / N);
        double t = (stat1.avgRuntime - stat2.avgRuntime) / s_x1x2;
        System.out.printf("t value for runtime: %f\n", t);

        s1 = stat1.varianceSteps;
        s2 = stat2.varianceSteps;
        N = Math.min(stat1.steps.size()-1, stat2.steps.size()-1);
        s_x1x2 = Math.sqrt( (s1 + s2) / N);
        t = (stat1.avgSteps - stat2.avgSteps) / s_x1x2;
        System.out.printf("t value for steps: %f\n", t);
    }

    /**
     * Wrapper for search with additional runtime measuring.
     * @param algo      search algorithm
     * @return          true if the solution is found; false otherwise.
     */
    static boolean run(ActorAgent.Algorithms algo) {
        harry.currentAlgo = algo;
        long start = System.nanoTime();
        boolean res = harry.find();
        long end = System.nanoTime();
        execTime = end-start;
        return res;
    }


    /**
     * Gives the console report on given search attempt.
     * @param success       success flag
     * @param showMap       should we print the map layout? flag
     */
    static void visualize(boolean success, boolean showMap) {
        System.out.println("\n=======================");
        System.out.printf("Used search algorithm: %s\n", harry.currentAlgo.toString());
        System.out.printf("Perception scenario: %d\n", harry.mooreN == -1 ? 2 : 1);
        System.out.printf("Outcome: ");
        if(!success) {
            System.out.printf("Lose\n");
            System.out.println("=======================");
            return;
        }
        else {
            System.out.printf("Win\n");
            System.out.printf("Number of steps to reach exit door: %d\n", harry.path.size()-1);
            System.out.printf("Execution time in microseconds: %.1f\n", (float) execTime/1000);
        }
        System.out.println("=======================");
        vis.addPath(harry.bookPath, true);
        vis.addPath(harry.exitPath, false);
        System.out.println("Path from start to exit:");

        while(!harry.path.isEmpty()) {
            System.out.printf("%s ", harry.path.pollFirst().toString());
        }
        if(showMap) {
            System.out.println("\nSolution on a map:");
            vis.printMap();
        }
        vis.clearPaths();
    }


    /**
     * Used for random map generation and user input validations. Check if the map layout complies with the assignment requirements.
     * @param bookLocation          location of a book
     * @param cloakLocation         location of a cloak
     * @param inspectors            array of inspectors
     * @return                      true if the positions are correct; if not, throws an exception
     * @throws MalformedInputException      the input does not satisfy the constraints of the assignment statement
     */
    static boolean validatePositions(Point bookLocation, Point cloakLocation, InspectorAgent[] inspectors) throws MalformedInputException {
        if(cenv.isCaughtInCell(bookLocation)               // book is not in inspectors' perception zone;
                || cenv.isCaughtInCell(cloakLocation)      // cloak is not in inspectors' perception zone
                )
            throw new MalformedInputException("Book or cloak must not be in inspector's perception zone");
        if(cenv.isCaughtInCell(harry.pos)) throw new MalformedInputException("Harry can not start in inspectors' perception zones");
        for(InspectorAgent inspector : inspectors) {
            if(cenv.getExitLocation().equals(inspector.pos)) // exit is not in inspectors' cells
                throw new MalformedInputException("Exit must not be in inspector's cell");
        }
        if(cenv.getExitLocation().equals(bookLocation)) // book and exit are not in the same cell
            throw new MalformedInputException("Book and exit must not be in the same cell");
        return true;
    }


    /**
     * Generates the map or fetches input from user.
     */
    static void init() {
        // input
        Random random = new Random();
        int failureCounter = 100;
        int bound = 9;
        ArrayList<Point> input = null;
        boolean inputOK = true;
        do {
            try {
                if(scenario <= 0) scenario = -1;
                inputOK = true;
                if(genMode == GenerationModes.MANUAL) System.out.println("Input coordinates of objects and agents and scenario:");
                if(genMode == GenerationModes.SEMIAUTO) System.out.println("Input scenario:");
                if(in == null)
                    in = new Scanner(System.in);
                input = new ArrayList<>(6);
                if(genMode == GenerationModes.MANUAL) {
                    String[] inputSplit = in.nextLine().split(" ");
                    if (inputSplit.length != 6) {
                        throw new MalformedInputException(String.format("Too few input data: expected 6 cells, got %d", inputSplit.length));
                    }
                    for (String s : inputSplit) {
                        input.add(Point.fromString(s));
                    }
                } else {
                    for (int i = 0; i < 6; i++) {
                        input.add(new Point(random.nextInt(bound), random.nextInt(bound)));
                    }
                }
                if(scenario <= 0) {
                    assert (genMode != GenerationModes.FULLAUTO);
                    scenario = Integer.parseInt(in.nextLine());
                    if (scenario != 1 && scenario != 2) {
                        scenario = 0;
                        throw new MalformedInputException("Invalid scenario number: must be either 1 or 2");
                    }
                }
                // intermediate assignment
                Point harryPos, catPos, filchPos, bookPos, cloakPos, exitPos;
                harryPos = input.get(0);
                filchPos = input.get(1);
                catPos = input.get(2);
                bookPos = input.get(3);
                cloakPos = input.get(4);
                exitPos = input.get(5);

                //init agents and environment
                InspectorAgent inspectorCat, inspectorFilch;
                InspectorAgent[] inspectors;
                inspectorCat = new InspectorAgent(catPos, 1);
                inspectorFilch = new InspectorAgent(filchPos, 2);

                inspectors = new InspectorAgent[]{inspectorCat, inspectorFilch};
                cenv = CellularEnvironment.getInstance(inspectors, bookPos,
                        exitPos, cloakPos, 9, 9);

                int harryMooreN = 1;
                if(scenario == 2) harryMooreN= -1;
                harry = new ActorAgent(harryPos, harryMooreN, cenv);
                vis = new Visualizer(cenv, harry);

                validatePositions(bookPos, cloakPos, inspectors);

                System.out.println("Map corresponding to input:");
                System.out.printf("Legend: Harry (👓) at %s, Filch (😡) at %s, Cat (😾) at %s, inspectors' perceptions zones are -\n" +
                        "\tbook (🕮) at %s, exit (🚪) at %s, cloak (👻) at %s\n", harryPos, inspectorFilch.pos, inspectorCat.pos, bookPos, exitPos, cloakPos);
                vis.makeInitialMap();
                vis.printMap();
            } catch(MalformedInputException miEx) {
                if(--failureCounter <= 0){
                    System.out.println("Too many incorrect input attempts, aborting...");
                    System.exit(1);
                }
                if(genMode == GenerationModes.MANUAL || (genMode == GenerationModes.SEMIAUTO && scenario == 0))
                    System.out.printf("Malformed input: %s\n", miEx.getMessage());
                inputOK = false;
                continue;
            }
        } while (!inputOK);

    }

}

class Visualizer {
    CellularEnvironment cenv;
    ActorAgent harry;
    String[][] map;
    static String arrowsVar1 = "🡠🡢🡡🡣🡤🡥🡦🡧";
    static String arrowsVar2 = "🠹🠸🠻🠺⬉⬈⬊⬋";
    public Visualizer(CellularEnvironment cenv, ActorAgent harry) {
        this.cenv = cenv;
        this.harry = harry;
        map = new String[cenv.worldRows][cenv.worldColumns];
    }

    public void makeInitialMap() {
        Point point = new Point(cenv.worldRows-1, 0); // top-right corner of map
        for(int i = cenv.worldRows-1; i >= 0; i--) {
            for(int j = 0; j < cenv.worldColumns; j++) {
                point.moveTo(i, j);
                if(point.equals(cenv.getBookLocation())) {
                    map[i][j] = "🕮";
                }
                else if(point.equals(cenv.getCloakLocation())){
                    map[i][j] = "👻";
                }
                else if(point.equals(cenv.getExitLocation())){
                    map[i][j] = "🚪";
                }
                else if(point.equals(cenv.getInspectors()[0].pos)) {
                    map[i][j] = "😾";
                }
                else if(point.equals(cenv.getInspectors()[1].pos)) {
                    map[i][j] = "😡";
                }
                else if(point.equals(harry.pos)) {
                    map[i][j] = "👓";
                }
                else if(cenv.isCaughtInCell(point)) {
                    map[i][j] = "-";
                }
                else {
                    map[i][j] = "□";
                }
            }
        }
    }

    /**
     * Writes the pseudographical representation of the map to console.
     */
    public void printMap() {
        for(int i = cenv.worldRows-1; i >= 0; i--) {
            System.out.printf("%d\t", i);
            for (int j = 0; j < cenv.worldColumns; j++) {
                System.out.printf("%s\t", map[i][j]);
            }
            System.out.printf("\n");
        }
        System.out.printf("\t");
        for(int i = 0; i < cenv.worldColumns; i++)
            System.out.printf("%d\t", i);
        System.out.printf("\n");
    }

    /**
     * Clears the map to prepare for next solution output.
     */
    public void clearPaths() {
        for(int i = 0; i < 9; i++){
            for(int j = 0; j < 9; j++){
                String tmp = map[i][j];
                if(tmp.length() > 1){
                    tmp = tmp.substring(1);
                }
               if(arrowsVar1.contains(tmp) || arrowsVar2.contains(tmp)) {
                   map[i][j] = "□";
               }
            }
        }
    }

    /**
     * Adds path on the map.
     * @param p             deque containing the path
     * @param style         thin arrows for start->book (or start->cloak->book) and bold ones for book->exit
     */
    public void addPath(ArrayDeque<Point> p, boolean style) {
        ArrayDeque<Point> path = p.clone();
        Point prev, next;
        prev = path.pollFirst();
        next = path.pollFirst();
        String[][] arrows = { {"🡧", "🡠", "🡤"}, { "🡣", "*", "🡡"}, {"🡦","🡢","🡥"}};
        String[][] arrowsBold = { {"⬋", "🠸", "⬉"}, { "🠻", "*", "🠹"}, {"⬊","🠺","⬈"}};
        while(next != null && prev != null) { //
            if(prev.equals(cenv.getBookLocation()))
                style = false;
            String s;
            int dx = next.x - prev.x;
            int dy = next.y - prev.y;
            if(style)
                s = arrows[dy+1][dx+1];
            else
                s = arrowsBold[dy+1][dx+1];
            map[prev.x][prev.y] = s;
            prev = next;
            next = path.pollFirst();
        }
        if (!style && prev != null)
            map[prev.x][prev.y] = "🚪";
        map[cenv.getBookLocation().x][cenv.getBookLocation().y] = "🕮";
        System.out.printf("\n");
    }
}
